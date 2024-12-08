import re
import spacy
from spacy.language import Language
from pdb import set_trace

def split_on_punct(doc):
    """
    Default punctuation-based SBD

    :param doc:
    :return:
    """
    start = 0
    seen_period = False
    for i, word in enumerate(doc):
        if seen_period and not word.is_punct:
            yield doc[start : word.i]
            start = word.i
            seen_period = False

        elif word.text in [".", "!", "?"]:
            seen_period = True
    if start < len(doc):
        yield doc[start : len(doc)]


def split_on_rgx(sentences, doc, rgx, threshold=250, sent_match=None, debug=False):
    """
    Match split tokens using provided regex.

    :param sentences:
    :param doc:
    :param rgx:
    :param threshold:
    :return:
    """
    splits = []
    for sent in sentences:
        if (
            len(sent.text) >= threshold
            and not sent_match
            or (len(sent.text) >= threshold and sent_match(sent))
        ):
            idxs = (
                [sent[0].i]
                + [word.i for word in sent if rgx.search(word.text)]
                + [sent[-1].i + 1]
            )
            idxs = sorted(list(set(idxs)))
            for i in range(len(idxs) - 1):
                splits.append(doc[idxs[i] : idxs[i + 1]])
        else:
            splits.append(sent)
    return splits


def split_on_phrase_rgx(sentences, doc, rgx, threshold=250):
    """
    Split sentence on phrase regex

    :param sentences:
    :param doc:
    :param rgx:
    :param threshold:
    :return:
    """
    splits = []
    for sent in sentences:
        matches = re.findall(rgx, sent.text)
        if len(sent.text) >= threshold and matches:
            offset = sent[0].idx
            # split up sentence
            m_idxs = set()
            for m in matches:
                m_idxs.add(sent.text.index(m) + offset)

            idxs = [sent[0].i]
            idxs += [word.i for word in sent if word.idx in m_idxs]
            idxs += [sent[-1].i + 1]

            idxs = sorted(list(set(idxs)))
            for i in range(len(idxs) - 1):
                splits.append(doc[idxs[i] : idxs[i + 1]])
        else:
            splits.append(sent)
    return splits


def merge_sentences(doc, sents, merge_terms):
    """
    Use a collection of bigrams (either from a dictionary of bigrams,
    corpus association weights, etc.) to define word pairs that cannot
    be split across sentences.

    TODO: Clean this up!

    :param doc:
    :param idxs:
    :param merge_terms:
    :return:
    """
    # terms that can never end a sentence
    non_terminals = {
        ",",
        "-",
        "(",
        "=",
        "/",
        "mrs.",
        "mr.",
        "ms.",
        "i.e.",
        ":",
        "dr.",
        "at",
        "with",
        "and",
        "the",
        "is",
        "s/p",
    }

    non_terminals_rgx = "(" + "|".join(map(re.escape, non_terminals)) + ")$"

    # word indices
    sequences = [[word.i for word in sent if word.text.strip()] for sent in sents]
    sequences = [idxs for idxs in sequences if len(idxs) != 0]

    stack = [sequences.pop(0)]
    for seq in sequences:
        i = stack[-1][-1]
        j = seq[0]
        text = re.sub(r"""\s{2,}|\n""", " ", doc[i : j + 1].text).lower().strip()

        # prior sentence
        prior = doc[stack[-1][0] : j]

        if re.search(non_terminals_rgx, prior.text, re.I):
            stack[-1].extend(seq)
        elif (merge_terms is not None and text in merge_terms) or doc[
            i
        ].text in non_terminals:
            stack[-1].extend(seq)
        else:
            stack.append(seq)

    # replace missing indices (whitespace)
    end = 0
    sentences = []
    stack[0] = [0] + stack[0]
    for i in range(len(stack) - 1):
        s = list(sorted(set(stack[i] + [stack[i + 1][0]])))
        sentences.append(doc[min(s) : max(s)])
        end = max(s)

    if doc[end:]:
        sentences.append(doc[end:])

    return sentences


def ct_sbd_min_rules(doc, merge_terms=None, max_sent_len=None):
    """Treat 3 contiguous whitespaces as a newline delimitter and split
    on punctuation as per a default punctuation rule-based splitter.

    Args:
        doc (_type_): _description_
        merge_terms (_type_, optional): _description_. Defaults to None.
        max_sent_len (_type_, optional): _description_. Defaults to None.

    Yields:
        _type_: _description_
    """
    sents = [sent for sent in split_on_punct(doc)]
    sents = split_on_rgx(sents, doc, re.compile("\s{3,}"), threshold=1)

    # combine sentences based on a list terms that cannot split
    merge_terms = {} if not merge_terms else merge_terms
    sents = merge_sentences(doc, sents, merge_terms)
    for s in sents:
        yield s


def ct_sbd_rules(doc, merge_terms=None, max_sent_len=None):
    """
    Split sentences if they don't meet certain char length thresholds.
    This splits on 3 and 2 character whitespace tokens and bulleted lists.

    :param doc:
    :param threshold:
    :return:
    """
    merge_terms = {} if not merge_terms else merge_terms

    sents = [sent for sent in split_on_punct(doc)]
    sents = split_on_rgx(sents, doc, re.compile("\s{2,}"), threshold=250)
    sents = split_on_rgx(
        sents,
        doc,
        re.compile("\s{1,}"),
        threshold=100,
        sent_match=lambda x: x.text.count(":") > 2,
    )
    sents = split_on_rgx(sents, doc, re.compile("[•¿](?![CF])"), threshold=10)
    sents = split_on_rgx(sents, doc, re.compile("\s{3,}"), threshold=0, debug=True)

    # combine sentences based on a list terms that cannot split
    sents = merge_sentences(doc, sents, merge_terms)

    # force sentences to have a max length
    # if max_sent_len:
    #     splits = []
    #     for s in sents:
    #         idxs = [word.i for word in s]
    #         if len(idxs) > max_sent_len:
    #             parts = list(toolz.partition_all(max_sent_len, idxs))
    #             for p in parts:
    #                 seq = doc[p[0] : p[-1] + 1]
    #                 splits.append(seq)
    #         else:
    #             seq = doc[idxs[0] : idxs[-1] + 1]
    #             splits.append(seq)

    #     sents = splits

    for s in sents:
        yield s


# @Language.component(name="clinical_text_heavy_sbd")
# def ct_sentence_boundaries(doc):
#     sents = list(ct_sbd_rules(doc))
#     start_token_ids = [s[0].idx for s in sents]
#     for token in doc:
#         token.is_sent_start = True if token.idx in start_token_ids else False
#     return doc


@Language.component(name="clinical_text_light_sbd")
def ct_sentence_boundaries(doc):
    sents = list(ct_sbd_min_rules(doc))
    start_token_ids = [s[0].idx for s in sents]
    for token in doc:
        token.is_sent_start = True if token.idx in start_token_ids else False
    return sents