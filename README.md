# JParaCrawl Fine-tuning Example
This repository includes an example usage of JParaCrawl pre-trained Neural Machine Translation (NMT) models.  
Our goal is to train (fine-tune) the domain-adapted NMT model in a few hours.

We wrote this document as beginner-friendly so that many people can try NMT experiments.
Thus, some parts might be too easy or redundant for experts.

In this example, we focus on fine-tuning the pre-trained model for KFTT corpus, which contains Wikipedia articles related to Kyoto.
We prepared two examples, English-to-Japanese and Japanese-to-English.
We recommend you to try the English-to-Japanese example if you are fluent Japanese speaker, otherwise Japanese-to-English since we expect you to read the MT output.
In the following, we use the Japanese-to-English example.


## JParaCrawl
JParaCrawl is the largest publicly available English-Japanese parallel corpus created by NTT.
In this example, we will fine-tune the model pre-trained on JParaCrawl.

For more details about JParaCrawl, visit the official web site.  
http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/


## Requirements
This example uses the followings.
- Python 3
- [PyTorch](https://pytorch.org/)
- [fairseq](https://github.com/pytorch/fairseq)
- [sentencepiece](https://github.com/google/sentencepiece)
- [MeCab](https://taku910.github.io/mecab/) with IPA dic
- NVIDIA GPU with CUDA

For fairseq, we recommend to use the same version as we pre-trained the model.
```
$ cd fairseq
$ git checkout c81fed46ac7868c6d80206ff71c6f6cfe93aee22
```

### Docker
We prepared the Docker container that was already installed the requisites.
Use the following commands to run.
Note that you can change `~/jparacrawl-experiments` to the path you want to store the experimental results.
This will be connected to the container as `/host_disk`.
``` sh
$ docker pull morinoseimorizo/jparacrawl-fairseq
$ docker run -it --gpus 1 -v ~/jparacrawl-experiments:/host_disk morinoseimorizo/jparacrawl-fairseq bash
```


## Prepare the data
First, you need to prepare the corpus and pre-trained model.

``` sh
$ cd /host_disk
$ git clone https://github.com/MorinoseiMorizo/jparacrawl-finetune.git   # Clone the repository.
$ cd jparacrawl-finetune
$ ./get-data.sh   # This script will download KFTT and sentencepiece model for pre-processing the corpus.
$ ./preprocess.sh   # Split the corpus into subwords.
$ cp ./ja-en/*.sh ./   # If you try the English-to-Japanese example, use en-ja directory instead.
$ ./get-model.sh   # Download the pre-trained model.
```

These commands will download the KFTT corpus and the pre-trained NMT model.
Then will tokenize the corpus to subwords with the provided sentencepiece models.
The subword tokenized corpus is located at `./corpus/spm`.
``` sh
$ head -n 2 corpus/spm/kyoto-train.en
▁Known ▁as ▁Se s shu ▁( 14 20 ▁- ▁150 6) , ▁he ▁was ▁an ▁ink ▁painter ▁and ▁Zen ▁monk ▁active ▁in ▁the ▁Muromachi ▁period ▁in ▁the ▁latter ▁half ▁of ▁the ▁15 th ▁century , ▁and ▁was ▁called ▁a ▁master ▁painter .
▁He ▁revolutionize d ▁the ▁Japanese ▁ink ▁painting .
```

You can see that a word is tokenized into several subwords.
We use subwords to reduce the vocabulary size and express a low-frequent word as a combination of subwords.
For example, the word `revolutionized` is tokenized into `revolutionize` and `d`.


## Decoding with pre-trained NMT models
Before fine-tuning experiments, let's try to decode (translate) a file with the pre-trained model to see how the current model works.
We prepared `decode.sh` that decodes the KFTT test set with the pre-trained NMT model.
``` sh
$ ./decode.sh
```

### Evaluation
We can automatically evaluate the translation results by comparing reference translations.
Here, we use [BLEU](https://www.aclweb.org/anthology/P02-1040/) scores, which is the most used evaluation matrix in the MT community.
The script automatically calculates the BLEU score and save it to `decode/test.log`.
BLEU scores ranges 0 to 100, so this result is somewhat low.
``` sh
$ cat decode/test.log
BLEU+case.mixed+numrefs.1+smooth.exp+tok.intl+version.1.4.2 = 14.2 50.4/22.0/11.2/5.9 (BP = 0.868 ratio = 0.876 hyp_len = 24351 ref_len = 27790)
```

It is also important to check outputs as well as BLEU scores.
Input and output files are located on `./corpus/kftt-data-1.0/data/orig/kyoto-test.ja` and `./decode/kyoto-test.ja.true.detok`.
```
$ head -n4 ./corpus/kftt-data-1.0/data/orig/kyoto-test.ja
InfoboxBuddhist
道元（どうげん）は、鎌倉時代初期の禅僧。
曹洞宗の開祖。
晩年に希玄という異称も用いた。。

$ head -n4 ./decode/kyoto-test.ja.true.detok
InfoboxBuddhist
Dogen is a Zen monk from the early Kamakura period.
The founder of the Soto sect.
In his later years, he also used the heterogeneous name "Legend".
```
This is just an example so the result may vary.

You can also find the reference translations at `./corpus/kftt-data-1.0/data/orig/kyoto-test.en`.
```
$ head -n4 ./corpus/kftt-data-1.0/data/orig/kyoto-test.en
Infobox Buddhist
Dogen was a Zen monk in the early Kamakura period.
The founder of Soto Zen
Later in his life he also went by the name Kigen.
```

The current model mistranslated the name "Kigen" to "Legend" at line 4.
Also, "heterogeneous" is not an appropriate translation.
Let's see how this could be improved by fine-tuning.


## Fine-tuning on KFTT corpus
Now, let's move to fine-tuning.
By fine-tuning, the model will adapt to the specific domain, KFTT.
Thus, we can expect the translation accuracy improves.

Following scripts will fine-tune the pre-trained model with the KFTT training set.
``` sh
$ nohup ./fine-tune_kftt_fp32.sh &> fine-tune.log &
$ tail -f fine-tune.log
```

Modern GPUs can use [mixed-precision training](https://arxiv.org/abs/1710.03740) that make use of Tensor Cores, which can compute half-precision floating-point faster.
If you want to use this feature, run `fine-tune_kftt_mixed.sh` instead of `fine-tune_kftt_fp32.sh` with Volta or later generations GPUs such as Tesla V100 or Geforce RTX 2080 Ti GPUs.

Training will take several hours to finish.
We tested on single RTX 2080Ti GPU with mixed-precision training and it finished in two hours.
Training time drastically differs based on the environment, so it may take a few more hours.

### Evaluation
Once it finished, you can find the BLEU score on the `models/fine-tune/test.log`.
You can see the BLEU score is greatly improved by fine-tuning.
``` sh
$ cat models/fine-tune/test.log
BLEU+case.mixed+numrefs.1+smooth.exp+tok.intl+version.1.4.2 = 26.4 57.8/31.7/20.1/13.5 (BP = 0.992 ratio = 0.992 hyp_len = 27572 ref_len = 27790)
```

Translated text is on `./models/fine-tune/kyoto-test.ja.true.detok`.
``` sh
$ head -n4 models/fine-tune/kyoto-test.ja.true.detok
Nickel buddhist
Dogen was a Zen priest in the early Kamakura period.
He was the founder of the Soto sect.
In his later years, he also used another name, Kigen.
```
The fine-tuned model could correctly translate line 4.


## Conclusion
In this document, we described how to use the pre-trained model and fine-tune it with KFTT.
By fine-tuning, we can obtain the domain-specific NMT model with a low computational cost.


## Next steps
We listed some examples to go further for NMT beginners.
- Looking into the provided scripts and find what commands are used.
- Try to translate your documents with the pre-trained and fine-tuned models.
    - You need to edit `decode.sh`.
    - See how well the model works.
- Try fine-tuning with other English-Japanese parallel corpora.
    - You can find the corpora from:
        - [OPUS](http://opus.nlpl.eu/)
        - [A list created by Prof. Neubig.](http://www.phontron.com/japanese-translation-data.php)
    - You need to tokenize it to subwords first.
        - Modify `preprocess.sh`.


## Further reading
- NMT architectures
    - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
    - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Subwords
    - [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
    - [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- Corpora
    - JParaCrawl: A Large Scale Web-Based English-Japanese Parallel Corpus
    - [The Kyoto Free Translation Task (KFTT)](http://www.phontron.com/kftt/)
- Tools
    - [fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/abs/1904.01038)


## Contact
Please send an issue on GitHub or contact us by email.  

NTT Communication Science Laboratories  
Makoto Morishita  
jparacrawl-ml -a- hco.ntt.co.jp  