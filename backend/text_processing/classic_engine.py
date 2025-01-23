import contextlib
import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from collections import namedtuple
from backend.text_processing import parsing, emphasis
from backend.text_processing.textual_inversion import EmbeddingDatabase
from backend import memory_management

PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
last_extra_generation_params = {}


@dataclass
class GenerationState:
    """Track generation parameters in a thread-safe way."""
    textual_inversions: Dict[str, str] = field(default_factory=dict)
    emphasis_method: Optional[str] = None
    
    def update_from_embeddings(self, embeddings: Dict[str, Any], embedding_key: str):
        """Update state from used embeddings."""
        for name in embeddings:
            print(f'[Textual Inversion] Used Embedding [{name}] in CLIP of [{embedding_key}]')
            clean_name = name.replace(":", "").replace(",", "")
            self.textual_inversions[clean_name] = embedding_key

    def update_from_texts(self, texts: List[str], emphasis_name: str):
        """Update state from processed texts."""
        if any("(" in text or "[" in text for text in texts):
            self.emphasis_method = emphasis_name

    def get_parameters(self) -> Dict[str, str]:
        """Get current generation parameters."""
        params = {}
        if self.textual_inversions:
            params["TI"] = ", ".join(self.textual_inversions.keys())
        if self.emphasis_method and self.emphasis_method != "Original":
            params["Emphasis"] = self.emphasis_method
        return params


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class SafeEmbeddingInjector(torch.nn.Module):
    """Safe replacement for CLIPEmbeddingForTextualInversion with explicit validation."""
    
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight
        
    def _validate_embedding(self, emb: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Validate and prepare embedding tensor."""
        if len(emb.shape) != 1:
            raise ValueError(f"Expected 1D embedding, got shape {emb.shape}")
        if emb.shape[0] > target_shape[0]:
            print(f"Warning: Truncating embedding from {emb.shape[0]} to {target_shape[0]}")
            emb = emb[:target_shape[0]]
        return emb
        
    def _inject_embedding(self, tensor: torch.Tensor, offset: int, 
                         embedding: torch.Tensor) -> torch.Tensor:
        """Safely inject embedding into tensor at offset."""
        available_space = tensor.shape[0] - offset - 1
        if available_space <= 0:
            return tensor
            
        emb = self._validate_embedding(embedding, tensor[offset+1:].shape)
        emb_len = min(available_space, emb.shape[0])
        
        # Create new tensor instead of in-place modification
        return torch.cat([
            tensor[:offset + 1],
            emb[:emb_len],
            tensor[offset + 1 + emb_len:]
        ], dim=0)

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None
        
        inputs_embeds = self.wrapped(input_ids)
        
        if not batch_fixes or max(len(x) for x in batch_fixes) == 0:
            return inputs_embeds
            
        # Process each sample with validation
        processed_tensors = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                # Get appropriate embedding vector
                emb = embedding.vec.get(self.textual_inversion_key, embedding.vec) \
                    if isinstance(embedding.vec, dict) else embedding.vec
                    
                # Convert to correct device/dtype
                emb = emb.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                
                # Inject embedding safely
                tensor = self._inject_embedding(tensor, offset, emb)
            processed_tensors.append(tensor)
            
        return torch.stack(processed_tensors, dim=0)


class ClassicTextProcessingEngine:
    def __init__(
        self, text_encoder, tokenizer, chunk_length=75,
        embedding_dir=None, embedding_key='clip_l', embedding_expected_shape=768,
        emphasis_name="Original", text_projection=False, minimal_clip_skip=1,
        clip_skip=1, return_pooled=False, final_layer_norm=True
    ):
        self.embeddings = EmbeddingDatabase(tokenizer, embedding_expected_shape)

        if isinstance(embedding_dir, str):
            self._load_embeddings(embedding_dir)

        self.embedding_key = embedding_key
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.emphasis = emphasis.get_current_option(emphasis_name)()
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        # Replace token embedding with custom embedding layer
        self._replace_token_embedding()

        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(',</w>', None)
        self.token_mults = self._compute_token_multipliers(vocab)

        self.generation_state = GenerationState()

    def _load_embeddings(self, embedding_dir):
        """Helper function to load embeddings from directory."""
        self.embeddings.add_embedding_dir(embedding_dir)
        self.embeddings.load_textual_inversion_embeddings()

    def _replace_token_embedding(self):
        """Helper function to replace token embedding with custom embedding layer."""
        model_embeddings = self.text_encoder.transformer.text_model.embeddings
        model_embeddings.token_embedding = SafeEmbeddingInjector(
            model_embeddings.token_embedding,
            self.embeddings,
            textual_inversion_key=self.embedding_key
        )

    def _compute_token_multipliers(self, vocab):
        """Helper function to compute token multipliers for emphasis."""
        token_mults = {}
        for text, ident in vocab.items():
            mult = 1.0
            for c in text:
                if c == '(':
                    mult *= 1.1
                elif c == ')':
                    mult /= 1.1
            if mult != 1.0:
                token_mults[ident] = mult
        return token_mults

    def empty_chunk(self):
        """Create an empty prompt chunk with padding."""
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """Calculate the target prompt token count for batching."""
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Tokenize a list of texts without truncation."""
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def encode_with_transformers(self, tokens):
        """Encode tokens using the transformer model."""
        target_device = memory_management.text_encoder_device()

        # Ensure embeddings are on the correct device and dtype
        embeddings = self.text_encoder.transformer.text_model.embeddings
        embeddings.to(device=target_device, dtype=torch.float32)

        tokens = tokens.to(target_device)

        outputs = self.text_encoder.transformer(tokens, output_hidden_states=True)

        layer_id = -max(self.clip_skip, self.minimal_clip_skip)
        z = outputs.hidden_states[layer_id]

        if self.final_layer_norm:
            # Apply final layer norm
            z = self.text_encoder.transformer.text_model.final_layer_norm(z)

        if self.return_pooled:
            pooled_output = outputs.pooler_output
            if self.text_projection:
                pooled_output = self.text_encoder.transformer.text_projection(pooled_output)
            z.pooled = pooled_output

        return z

    def _find_best_split_point(self, chunk, max_distance=20):
        """
        Find the best position to split a chunk, preferring commas.
        
        Args:
            chunk: PromptChunk to analyze
            max_distance: Maximum distance from chunk length to consider for comma split
        
        Returns:
            int: Position to split at, or -1 if no good split point found
        """
        # Look for last comma within acceptable range
        chunk_len = len(chunk.tokens)
        min_pos = max(0, chunk_len - max_distance)
        
        i = next((i for i in range(chunk_len - 1, min_pos - 1, -1) if chunk.tokens[i] == self.comma_token), -1)
        return i + 1 if i != -1 else -1

    def _add_embedding_to_chunk(self, chunk, embedding, position, weight):
        """
        Add an embedding to a chunk, handling overflow cases.
        
        Returns:
            bool: True if embedding was added, False if chunk needs to be finalized
        """
        emb_len = int(embedding.vectors)
        if len(chunk.tokens) + emb_len > self.chunk_length:
            return False
            
        chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))
        chunk.tokens.extend([0] * emb_len)  # Placeholder tokens
        chunk.multipliers.extend([weight] * emb_len)
        return True

    def _finalize_chunk(self, chunk, chunks, is_last=False):
        """
        Finalize a chunk by padding and adding special tokens.
        """
        if not chunk.tokens and not is_last:
            return
            
        # Calculate required padding
        current_len = len(chunk.tokens)
        pad_len = self.chunk_length - current_len
        
        # Add padding if needed
        if pad_len > 0:
            chunk.tokens.extend([self.id_end] * pad_len)
            chunk.multipliers.extend([1.0] * pad_len)
            
        # Add start/end tokens
        chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
        chunks.append(chunk)

    def tokenize_line(self, line):
        """
        Tokenize a line of text into chunks, handling emphasis and embeddings.
        
        Args:
            line: Input text line
            
        Returns:
            tuple: (list of PromptChunk objects, total token count excluding special tokens)
        """
        # Parse and tokenize the input
        parsed = parsing.parse_prompt_attention(line)
        tokenized = self.tokenize([text for text, _ in parsed])
        
        chunks = []
        chunk = PromptChunk()
        last_comma = -1
        
        for tokens_list, (text, weight) in zip(tokenized, parsed):
            # Handle BREAK marker
            if text == 'BREAK' and weight == -1:
                self._finalize_chunk(chunk, chunks)
                chunk = PromptChunk()
                continue
                
            position = 0
            while position < len(tokens_list):
                # Check if current chunk is full
                if len(chunk.tokens) >= self.chunk_length:
                    # Try to split at a good position
                    split_point = self._find_best_split_point(chunk)
                    if split_point > 0:
                        # Split chunk at comma
                        new_chunk = PromptChunk()
                        new_chunk.tokens = chunk.tokens[split_point:]
                        new_chunk.multipliers = chunk.multipliers[split_point:]
                        chunk.tokens = chunk.tokens[:split_point]
                        chunk.multipliers = chunk.multipliers[:split_point]
                    
                    self._finalize_chunk(chunk, chunks)
                    chunk = new_chunk if split_point > 0 else PromptChunk()
                    last_comma = -1
                
                # Track comma positions
                token = tokens_list[position]
                if token == self.comma_token:
                    last_comma = len(chunk.tokens)
                
                # Handle embeddings
                embedding, embedding_length = self.embeddings.find_embedding_at_position(
                    tokens_list, position)
                    
                if embedding is not None:
                    if not self._add_embedding_to_chunk(chunk, embedding, position, weight):
                        self._finalize_chunk(chunk, chunks)
                        chunk = PromptChunk()
                        # Retry adding embedding to new chunk
                        self._add_embedding_to_chunk(chunk, embedding, position, weight)
                    position += embedding_length
                else:
                    # Add regular token
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
        
        # Finalize last chunk
        self._finalize_chunk(chunk, chunks, is_last=True)
        
        # Calculate total tokens excluding special tokens
        token_count = sum(len(c.tokens) - 2 for c in chunks)
        
        return chunks, token_count

    def process_texts(self, texts):
        """Process a list of texts into tokenized chunks."""
        token_count = 0
        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks
            batch_chunks.append(chunks)
        return batch_chunks, token_count

    def __call__(self, texts):
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max(len(chunks) for chunks in batch_chunks)

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [chunk.tokens for chunk in batch_chunk]
            multipliers = [chunk.multipliers for chunk in batch_chunk]
            self.embeddings.fixes = [chunk.fixes for chunk in batch_chunk]

            # Collect used embeddings
            for fixes in self.embeddings.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        # Update generation state
        self.generation_state.update_from_embeddings(used_embeddings, self.embedding_key)
        self.generation_state.update_from_texts(texts, self.emphasis.name)

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, tokens_list: List[List[int]], 
                      multipliers_list: List[List[float]]) -> torch.Tensor:
        """Process tokens with safer tensor operations."""
        tokens = torch.tensor(tokens_list, dtype=torch.long)
        
        # Replace padding tokens after end token
        for batch_idx, token_seq in enumerate(tokens):
            try:
                end_idx = token_seq.tolist().index(self.id_end)
                # Create new tensor instead of in-place modification
                padding_mask = torch.arange(len(token_seq)) > end_idx
                tokens[batch_idx] = torch.where(
                    padding_mask, 
                    torch.full_like(token_seq, self.id_pad),
                    token_seq
                )
            except ValueError:
                continue
                
        z = self.encode_with_transformers(tokens)
        pooled = getattr(z, 'pooled', None)

        # Apply emphasis with explicit tensor operations
        emphasis_multipliers = torch.tensor(
            multipliers_list, 
            device=z.device, 
            dtype=z.dtype
        )
        
        # Set up emphasis inputs
        self.emphasis.tokens = tokens_list
        self.emphasis.multipliers = emphasis_multipliers
        self.emphasis.z = z.clone()  # Work on a copy
        
        # Apply emphasis and get result
        self.emphasis.after_transformers()
        z = self.emphasis.z
        
        if pooled is not None:
            z.pooled = pooled
            
        return z

    def get_prompt_lengths_on_ui(self, prompt):
        """Get the token count and target token count for a given prompt."""
        _, token_count = self.process_texts([prompt])
        target_count = self.get_target_prompt_token_count(token_count)
        return token_count, target_count
