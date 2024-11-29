import torch

from collections import namedtuple
from backend.text_processing import parsing, emphasis
from backend import memory_management


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, emphasis_name="Original", min_length=256):
        super().__init__()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer

        self.emphasis = emphasis.get_current_option(emphasis_name)()
        self.min_length = min_length
        self.id_end = 1
        self.id_pad = 0

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            multiplier_factor = 1.1

            # Use a more efficient calculation for nested brackets
            open_brackets = text.count('[') - text.count(']')
            open_parens = text.count('(') - text.count(')')

            mult *= pow(multiplier_factor, open_parens)
            mult /= pow(multiplier_factor, open_brackets)

            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def encode_with_transformers(self, tokens):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)

        # Optimize batch processing
        batch_size = tokens.shape[0]
        if batch_size > 1:
            # Process in smaller batches if input is large
            max_batch = 8
            z_list = []
            for i in range(0, batch_size, max_batch):
                batch = tokens[i:i + max_batch]
                z_list.append(self.text_encoder(input_ids=batch))
            z = torch.cat(z_list, dim=0)
        else:
            z = self.text_encoder(input_ids=tokens)

        return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        # Pre-allocate arrays for better performance
        chunk.tokens = []
        chunk.multipliers = []

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            if chunk.tokens:  # Only process if there are tokens
                chunk.tokens.append(self.id_end)
                chunk.multipliers.append(1.0)
                current_chunk_length = len(chunk.tokens)
                token_count += current_chunk_length

                remaining_count = self.min_length - current_chunk_length
                if remaining_count > 0:
                    chunk.tokens.extend([self.id_pad] * remaining_count)
                    chunk.multipliers.extend([1.0] * remaining_count)

                chunks.append(chunk)

            chunk = PromptChunk()
            chunk.tokens = []
            chunk.multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts):
        zs = []
        cache = {}

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                if not chunks:
                    continue

                # Optimize padding by pre-allocating arrays
                max_tokens = max(len(chunk.tokens) for chunk in chunks)
                line_z_values = []

                for chunk in chunks:
                    remaining_count = max_tokens - len(chunk.tokens)
                    if remaining_count > 0:
                        # Pre-allocate arrays and fill in one operation
                        tokens = torch.full((max_tokens,), self.id_pad, dtype=torch.long)
                        multipliers = torch.ones(max_tokens, dtype=torch.float)
                        tokens[:len(chunk.tokens)] = torch.tensor(chunk.tokens)
                        multipliers[:len(chunk.multipliers)] = torch.tensor(chunk.multipliers)
                    else:
                        tokens = torch.tensor(chunk.tokens)
                        multipliers = torch.tensor(chunk.multipliers)

                    z = self.process_tokens([tokens.tolist()], [multipliers.tolist()])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
