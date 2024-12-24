import math
import torch

from collections import namedtuple
from backend.text_processing import parsing, emphasis
from backend.text_processing.textual_inversion import EmbeddingDatabase
from backend import memory_management

PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
last_extra_generation_params = {}


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class CLIPEmbeddingForTextualInversion(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None  # Reset fixes after use

        inputs_embeds = self.wrapped(input_ids)

        if not batch_fixes or max(len(x) for x in batch_fixes) == 0:
            return inputs_embeds

        # Process each sample in the batch
        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec.get(self.textual_inversion_key, embedding.vec) \
                    if isinstance(embedding.vec, dict) else embedding.vec
                emb = emb.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                # Efficiently replace tensor slices
                tensor = torch.cat([
                    tensor[:offset + 1],
                    emb[:emb_len],
                    tensor[offset + 1 + emb_len:]
                ], dim=0)
            vecs.append(tensor)

        return torch.stack(vecs, dim=0)


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

    def _load_embeddings(self, embedding_dir):
        """Helper function to load embeddings from directory."""
        self.embeddings.add_embedding_dir(embedding_dir)
        self.embeddings.load_textual_inversion_embeddings()

    def _replace_token_embedding(self):
        """Helper function to replace token embedding with custom embedding layer."""
        model_embeddings = self.text_encoder.transformer.text_model.embeddings
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(
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

    def tokenize_line(self, line):
        """Tokenize a single line of text with attention to emphasis and embeddings."""
        parsed = parsing.parse_prompt_attention(line)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        for tokens_list, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                self._finalize_chunk(chunk, chunks)
                chunk = PromptChunk()
                continue

            position = 0
            tokens = tokens_list
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                if len(chunk.tokens) == self.chunk_length:
                    if last_comma != -1 and len(chunk.tokens) - last_comma <= 20:
                        # Split chunk at last comma
                        break_location = last_comma + 1
                        self._split_chunk_at(chunk, break_location, chunks)
                        chunk = PromptChunk()
                    else:
                        self._finalize_chunk(chunk, chunks)
                        chunk = PromptChunk()
                    last_comma = -1

                # Handle embeddings
                embedding, embedding_length = self.embeddings.find_embedding_at_position(tokens, position)
                if embedding is not None:
                    emb_len = int(embedding.vectors)
                    if len(chunk.tokens) + emb_len > self.chunk_length:
                        self._finalize_chunk(chunk, chunks)
                        chunk = PromptChunk()
                    chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))
                    chunk.tokens.extend([0] * emb_len)
                    chunk.multipliers.extend([weight] * emb_len)
                    position += embedding_length
                else:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1

        if chunk.tokens or not chunks:
            self._finalize_chunk(chunk, chunks, is_last=True)

        token_count = sum(len(c.tokens) - 2 for c in chunks)  # Exclude start/end tokens
        return chunks, token_count

    def _finalize_chunk(self, chunk, chunks, is_last=False):
        """Helper function to finalize and pad a chunk."""
        # Pad chunk tokens and multipliers
        to_add = max(0, self.chunk_length - len(chunk.tokens))
        if to_add > 0:
            chunk.tokens.extend([self.id_end] * to_add)
            chunk.multipliers.extend([1.0] * to_add)
        # Add start and end tokens
        chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
        chunks.append(chunk)

    def _split_chunk_at(self, chunk, break_location, chunks):
        """Helper function to split a chunk at a specified location."""
        # Split tokens and multipliers
        reloc_tokens = chunk.tokens[break_location:]
        reloc_mults = chunk.multipliers[break_location:]
        chunk.tokens = chunk.tokens[:break_location]
        chunk.multipliers = chunk.multipliers[:break_location]
        # Finalize the current chunk
        self._finalize_chunk(chunk, chunks)
        # Start a new chunk with the relocated tokens
        new_chunk = PromptChunk()
        new_chunk.tokens = reloc_tokens
        new_chunk.multipliers = reloc_mults
        chunks.append(new_chunk)

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

        # Update global parameters for used embeddings
        self._update_last_extra_generation_params(used_embeddings, texts)

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def _update_last_extra_generation_params(self, used_embeddings, texts):
        """Helper function to update last generation parameters with used embeddings."""
        global last_extra_generation_params

        if used_embeddings:
            names = []
            for name in used_embeddings:
                print(f'[Textual Inversion] Used Embedding [{name}] in CLIP of [{self.embedding_key}]')
                names.append(name.replace(":", "").replace(",", ""))
            if "TI" in last_extra_generation_params:
                last_extra_generation_params["TI"] += ", " + ", ".join(names)
            else:
                last_extra_generation_params["TI"] = ", ".join(names)

        if any("(" in text or "[" in text for text in texts) and self.emphasis.name != "Original":
            last_extra_generation_params["Emphasis"] = self.emphasis.name

    def process_tokens(self, tokens_list, multipliers_list):
        """Process tokens and multipliers through the model."""
        tokens = torch.tensor(tokens_list, dtype=torch.long)

        # Replace padding tokens after end token
        for batch_idx, token_seq in enumerate(tokens):
            try:
                end_idx = token_seq.tolist().index(self.id_end)
                tokens[batch_idx, end_idx + 1:] = self.id_pad
            except ValueError:
                pass  # id_end not found; no action needed

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        # Apply emphasis
        self.emphasis.tokens = tokens_list
        self.emphasis.multipliers = torch.tensor(multipliers_list, device=z.device, dtype=z.dtype)
        self.emphasis.z = z
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
