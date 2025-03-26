import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import treebank
from nltk import CFG
from nltk.parse.chart import ChartParser
from nltk import PCFG

#
# nltk.download("popular")
# nltk.download('averaged_perceptron_tagger_eng')

# sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
#
# tagged = nltk.pos_tag(tokens) # PoS tagging
# print("PoS:",tagged)
#
# s = "Good muffins cost $3.88\nin New York. Please buy me two of them.\n\nThanks."
# tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
# output = tokenizer.tokenize(s)
# print(output)

#t = treebank.parsed_sents('wsj_0001.mrg')[0]
#t.draw()

#Exercise 1
def exercise_1():
    """Tokenization and PoS tagging for given sentences."""
    print("\n=== Exercise 1: Tokenization and PoS Tagging ===")
    sentences = [
        "The Jamaica Observer reported that Usain Bolt broke the 100m record.",
        "While hunting in Africa, I shot an elephant in my pajamas. How an elephant got into my pajamas I'll never know."
    ]
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)  # Tokenization
        pos_tags = nltk.pos_tag(tokens)  # PoS tagging

        print("\nSentence:", sentence)
        print("Tokens:", tokens)
        print("PoS Tags:", pos_tags)

#Exercise 2

def exercise_2():
    """Parse sentences using a CFG grammar."""
    print("\n=== Exercise 2: Parsing with CFG ===")

    # Define a CFG grammar
    grammar = CFG.fromstring("""
        S -> NP VP
        NP -> Det N | Det N PP | 'John' | 'Alex' | 'the' N PP | 'the' N | 'a' N PP | 'a' N
        VP -> V NP | VP PP
        PP -> P NP
        Det -> 'my' | 'a' | 'the'
        N -> 'man' | 'telescope' | 'dog' | 'sandwich' | 'park'
        V -> 'saw' | 'kissed' | 'ate'
        P -> 'with' | 'in'
    """)

    sentences = [
        "John saw a man with my telescope",
        "Alex kissed the dog",
        "the man with the telescope ate a sandwich in the park"
    ]

    parser = ChartParser(grammar)
    for sentence in sentences:
        print(f"\nParsing: {sentence}")
        tokens = nltk.word_tokenize(sentence)
        trees = list(parser.parse(tokens))

        if not trees:
            print("No valid parse tree found.")
            continue

        for tree in trees:
            print(tree)
            tree.pretty_print()

        if len(trees) > 1:
            print("Ambiguity detected: The sentence has multiple valid interpretations.")

#Exercise 3

#The lexicons are “my”, “a”, “the”, “man”, “telescope”, “dog”, “sandwich”, “park”, “saw”, “kissed”, “ate”, “with”, “in”, “John”, “Alex”

#We can make a sentence like Alex saw John eat a sandwish:
#	The verb “saw” requires a noun phrase  as its object (e.g., “Alex saw a man”), but in this case, its object is “John eat a sandwich”, which is a full clause (NP + VP).
#	The grammar does not have a rule that allows a VP to contain another VP in this way.


#Exercise 4:

def exercise_5():
    pcfg1 = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
    Det -> 'the' [0.8] | 'a' [0.2]
    N -> 'man' [0.5] | 'telescope' [0.5]
    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
    V -> 'ate' [0.35] | 'saw' [0.65]
    PP -> P NP [1.0]
    P -> 'with' [0.61] | 'in' [0.39]
    """)

    print(pcfg1)
    text = "I saw the man with a telescope"
    text_tokens = nltk.word_tokenize(text)
    viterbi_parser = nltk.ViterbiParser(pcfg1, trace=0)
    trees = viterbi_parser.parse(text_tokens)
    for tree in trees:
        tree.pretty_print()
        print(tree)
        tree.draw()

#trace (int) – The level of tracing that should be used when parsing a text. 0 will generate no tracing output; and higher numbers will produce more verbose tracing output.
#trace = 0 = p=0.000104081
#trace = 1 =p=0.000104081
#trace = 2 =p=0.000104081
#It determines how much internal processing information the parser outputs while parsing a sentence.
#trace=0: No tracing (default). The parser runs silently and only returns the parse tree(s).
#trace=1: Minimal tracing, showing a basic step-by-step progression.
#trace=2: More detailed tracing, including probabilities and parsing decisions.
#trace=3: Verbose tracing, showing all parsing steps, including rule applications and intermediate states.


def exercise_6():
    """
    Function to learn a PCFG grammar from the NLTK treebank, parse a sentence using ViterbiParser,
    and print the parse tree along with its probability.

    :param trace_level: Verbosity level for parsing (1 or 2).
    """
    # Ensure necessary NLTK resources are available
    trace_level = 2

    nltk.download('treebank')
    nltk.download('punkt')

    productions = []
    S = nltk.Nonterminal('S')

    # Extract productions from the treebank corpus
    for f in treebank.fileids():
        for tree in treebank.parsed_sents(f):
            productions += tree.productions()

    # Induce PCFG grammar
    grammar = nltk.induce_pcfg(S, productions)

    # Print some productions
    print("First 25 Grammar Productions:")
    for p in grammar.productions()[:25]:
        print(p)

    # Initialize parser with trace parameter
    myparser = nltk.ViterbiParser(grammar, trace=trace_level)

    # Tokenize input text
    text = "the boy jumps over the board"
    mytokens = nltk.word_tokenize(text)

    # Parse the sentence
    parses = list(myparser.parse(mytokens))

    # Output parsing results
    if parses:
        best_parse = parses[0]
        print("\nParse Tree:")
        print(best_parse)
        print("\nProbability:", best_parse.prob())
    else:
        print("\nNo valid parse found.")
# def exercise_6():
#     pcfg1 = PCFG.fromstring("""
#         S -> NP VP [1.0]
#         NP -> Det N [0.6] | NP PP [0.1] | 'John' [0.15] | 'I' [0.15]
#         Det -> 'the' [0.8] | 'a' [0.2]
#         N -> 'man' [0.5] | 'telescope' [0.5]
#         VP -> VP PP [0.5] | V NP [0.4] | V [0.1]
#         V -> 'ate' [0.35] | 'saw' [0.65]
#         PP -> P NP [1.0]
#         P -> 'with' [0.61] | 'in' [0.39]
#         """)
#
#     print("Modified PCFG Grammar:\n", pcfg1)
#
#     text = "I saw the man with a telescope"
#     text_tokens = nltk.word_tokenize(text)
#
#     viterbi_parser = nltk.ViterbiParser(pcfg1)
#     trees = list(viterbi_parser.parse(text_tokens))
#
#     for tree in trees:
#         print("\nModified Parse Tree (Forced Alternative Interpretation):")
#         tree.pretty_print()
#         print(tree)
#         tree.draw()

def exercise_7():
    productions = []
    S = nltk.Nonterminal('S')
    for f in treebank.fileids():
        for tree in treebank.parsed_sents(f):
            productions += tree.productions()
    grammar = nltk.induce_pcfg(S, productions)
    for p in grammar.productions()[1:25]:
        print(p)
    myparser = nltk.ViterbiParser(grammar, 2)
    text = "the boy jumps over the board"
    mytokens = nltk.word_tokenize(text)
    myparsing, = myparser.parse(mytokens)
    print(myparsing)
    # Parse the sentence
    trees = list(myparser.parse(mytokens))

    # Display the parse tree in a nicely formatted way
    if trees:
        print("\nParsed Tree:\n")
        trees[0].pretty_print()


def test():
    import nltk
    from nltk.corpus import treebank
    productions = []
    S = nltk.Nonterminal('S')
    for f in treebank.fileids():
        for tree in treebank.parsed_sents(f):
            productions += tree.productions()
    grammar = nltk.induce_pcfg(S, productions)
    for p in grammar.productions()[1:25]:
        print(p)
    myparser = nltk.ViterbiParser(grammar, 1)
    text = "the boy jumps over the board"
    mytokens = nltk.word_tokenize(text)
    myparsing, = myparser.parse(mytokens)
    print(myparsing)

RUN_EXERCISE_1 = False
RUN_EXERCISE_2 = False
RUN_EXERCISE_5 = False
RUN_EXERCISE_6 = False
RUN_EXERCISE_7 = False
RUN_TEST = True

def main():
    print("\n=== NLTK Processing Script ===")

    if RUN_EXERCISE_1:
        exercise_1()

    if RUN_EXERCISE_2:
        exercise_2()

    if RUN_EXERCISE_5:
        exercise_5()

    if RUN_EXERCISE_6:
        print("Exercise 6")
        exercise_6()

    if RUN_EXERCISE_7:
        print("Exercise 7")
        exercise_7()

    if RUN_EXERCISE_7:
        print("Exercise Test")
        exercise_test()

if __name__ == "__main__":
    main()