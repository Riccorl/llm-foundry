import argparse
import psutil
import sentencepiece as spm
import transformers as tr
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model-prefix", type=str)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--character-coverage", type=float, default=0.9999)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--hf-class", type=str, default="LlamaTokenizer")
    args = parser.parse_args()

    start = time.time()
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type="bpe",
        byte_fallback=True,
        split_digits=True,
        num_threads=args.num_threads or psutil.cpu_count(logical=True),
    )
    end = time.time()
    print("Training took {} seconds".format(end - start))

    # assumes a llama tokenizer
    print("Saving HF tokenizer")
    hf_class = getattr(tr, args.hf_class)
    hf_tokenizer = hf_class(args.model_prefix + ".model")
    hf_tokenizer.save_pretrained(args.model_prefix + "-hf")
