import gzip
import logging
from abc import ABC
from pathlib import Path
from typing import Iterable, TypeVar

import pyknp
import pyknp_eventgraph as pyevg
import pytest
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Token
from textformatting import ssplit

from utils.utils import read_seed_lex

logger = logging.getLogger(__name__)


class Extractor(ABC):
    T = TypeVar("T")
    BLACK_LIST_CHARS = set("「」【】『』［］（）！？｜〜…：；＠＃＊／＾" + '@!?()"_/<>^#*:;[]')

    def __init__(self, data_file: str, seed_lex_file: str):
        self.data_file = Path(data_file)
        self.seed_lex_file = Path(seed_lex_file)

        self.seed_words = None

    def __call__(self, filter_by_seed: bool = False):
        self.seed_words = read_seed_lex(self.seed_lex_file)
        logger.info(f"Extract events from {self.data_file.absolute()}.")
        with gzip.open(self.data_file, "rt", errors="ignore") as f:
            for line in f:
                if line.strip() == "":
                    continue
                for event in filter(
                    self.is_valid, self.convert_line_to_events(line, filter_by_seed)
                ):
                    yield event

    def convert_line_to_events(self, line: str, filter_by_seed: bool) -> Iterable[T]:
        raise NotImplementedError

    def is_valid(self, event: T) -> bool:
        raise NotImplementedError

    def include_seed_word(self, text: str):
        return any(seed_word.text in text for seed_word in self.seed_words)


class JapaneseExtractor(Extractor):
    def __init__(self, data_file: str, seed_lex_file: str):
        super().__init__(data_file, seed_lex_file)
        self.knp = None

    def convert_line_to_events(
        self, line: str, filter_by_seed: bool = False
    ) -> Iterable[pyevg.Event]:
        if filter_by_seed and not self.include_seed_word(line):
            return
        for sent in ssplit(line.strip()):
            if len(set(sent) & self.BLACK_LIST_CHARS) > 0:
                continue
            if filter_by_seed and not self.include_seed_word(sent):
                continue
            if self.knp is None:
                self.knp = pyknp.KNP()
            try:
                evg = pyevg.EventGraph.build([self.knp.parse(sent)])
                if len(evg.events) > 0:
                    yield evg.events[-1]
            except Exception as e:
                logger.warning(e)
                continue

    def is_valid(self, event: pyevg.Event) -> bool:
        pred = event.pas.predicate.head
        # 文の最後の節でないなら除外．
        if event != event.sentence.events[-1]:
            return False
        # 動詞性の用言でないなら除外．
        if pred.features.get("体言", False):
            return False
        if "判定詞" in pred.features:
            return False
        if pred.features.get("主辞代表表記", "") == "有り難い/ありがたい":
            # 有難うございました
            return False
        if pred.features.get("用言", "") != "動":
            return False
        if pred.features.get("ト", False):
            return False
        # 可能表現・受動表現なら除外; 明らかに意志的でないため学習の必要はなく，かつ副詞の手がかりより強いためノイズになる．
        if pred.features.get("態", "") in {"受動", "可能", "受動|可能"}:
            return False
        pred_rep = pred.features.get("用言代表表記", "")
        if pred.features.get("可能表現", False) or "こと/こと+可能だ/かのうだ" in pred_rep:
            return False
        # モダリティを含むなら除外．モダリティは意志性と直交する意味情報．
        if len(event.features.modality) > 0:
            return False
        if any(m.hinsi == "助動詞" for m in pred.mrph_list()):
            return False
        if event.head != event.end:
            # e.g., 忘れず持ってくること。, お金は返ってくるはず。, 勝負に持ち込みたいところ。
            return False
        return True


class EnglishExtractor(Extractor):
    def __init__(self, data_file: str, seed_lex_file: str):
        super().__init__(data_file, seed_lex_file)
        self.nlp = None

    def convert_line_to_events(
        self, line: str, filter_by_seed: bool = False
    ) -> Iterable[Span]:
        if filter_by_seed and not self.include_seed_word(line):
            return
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        try:
            doc = self.nlp(line.strip())
        except Exception as e:
            logger.warning(e)
            return
        for sent in doc.sents:  # type: Span
            if filter_by_seed and not self.include_seed_word(sent.text):
                continue
            if len(set(sent.text) & self.BLACK_LIST_CHARS) > 0:
                continue
            yield sent

    def is_valid(self, event: Span) -> bool:
        if len(event.text) >= 100:
            return False

        if event.text[0] != event.text[0].upper():
            return False

        if event.text.startswith("Thank"):
            return False

        root = event.root  # type: Token

        # sentential complements
        cdeps = {"conj", "ccomp", "advcl"}
        if len([c for c in root.children if c.dep_ in cdeps]) > 0:
            return False

        # exclude if the predicate is not a verb
        be_verbs = ["be", "s", "’", "'", "’m", "'m", "’re", "'re"]
        if root.pos_ != "VERB" or root.lemma_ in be_verbs:
            return False

        if event.text.startswith("Thank"):
            return False

        # exclude question
        if event.text.endswith("?"):
            return False
        if event[0].lemma_ in {"do", "what", "when", "where", "why", "who", "how"}:
            return False

        # exclude imperative
        if sum(c.dep_ == "nsubj" for c in root.children) == 0:
            return False

        # exclude potential/passive
        # https://gist.github.com/armsp/30c2c1e19a0f1660944303cf079f831a
        modal_auxs = [
            # ability
            "can",
            "ca",  # can't
            "could",
            "may",
            "might",
            "shall",
            # willingness
            "will",
            "'ll",
            "’ll",  # X'll
            "wo",  # won't
            "would",
            "’d",
            "'d",
            # necessity
            "must",
            "should",
        ]
        for c in root.children:
            if c.dep_ == "aux" and c.lemma_ in modal_auxs:
                return False

        matcher = Matcher(self.nlp.vocab)
        matcher.add(
            "Passive",
            [
                [
                    {"DEP": "auxpass"},
                    {"DEP": "neg", "OP": "?"},
                    {"DEP": {"IN": ["aux", "advmod"]}, "OP": "*"},
                    {"DEP": "ROOT", "TAG": "VBN"},
                ]
            ],
        )
        matcher.add(
            "Modality",
            [
                [
                    {
                        "DEP": "ROOT",
                        "LEMMA": {
                            "IN": [
                                # necessity
                                "have",
                                "'ve",
                                "’ve",
                                "need",
                                "ought",
                                # volition
                                "want",
                                # certainty
                                "seem",
                                "appear",
                            ]
                        },
                    },
                    {"LEMMA": "to"},
                    {"TAG": "VB"},
                ],
                # volition
                [
                    {"LEMMA": {"IN": ["’d", "'d", "would"]}},
                    {"DEP": "ROOT", "LEMMA": "like"},
                    {"LEMMA": "to"},
                    {"TAG": "VB"},
                ],
            ],
        )
        matcher.add(
            "Future",
            [
                [
                    {"LEMMA": {"IN": be_verbs}},
                    {"DEP": "neg", "OP": "?"},
                    {"DEP": {"IN": ["aux", "advmod"]}, "OP": "*"},
                    {"DEP": "ROOT", "LOWER": "going"},
                    {"LEMMA": "to"},
                    {"TAG": "VB"},
                ]
            ],
        )

        if matcher(event):
            return False

        return True


@pytest.mark.parametrize(
    "text,is_valid",
    [
        ("I go to school.", True),
        ("I swim.", True),
        # adjective
        ("I am kind.", False),
        ("It is beautiful.", False),
        # copula
        ("I am a student.", False),
        ("I'm a student.", False),
        ("You are a student.", False),
        ("You're a student.", False),
        ("He is a student.", False),
        ("He's a student.", False),
        ("They are students.", False),
        ("They're students.", False),
        ("We are students.", False),
        ("We're students.", False),
        ("That is what I want.", False),
        # modality
        ("I can eat it.", False),
        ("I can't eat it.", False),
        ("I could eat it.", False),
        ("I couldn't eat it.", False),
        ("You may eat it.", False),
        ("He might eat it.", False),
        ("You shall not pass.", False),
        ("You will receive power.", False),
        ("You'll receive power.", False),
        ("You won't receive power.", False),
        ("You would receive power.", False),
        ("You must receive power.", False),
        ("You should receive power.", False),
        ("You have to receive power.", False),
        ("You need to receive power.", False),
        ("You ought to receive power.", False),
        ("I want to receive power.", False),
        ("I would like to receive power.", False),
        ("I'd like to receive power.", False),
        ("I’d like to receive power.", False),
        ("I want to receive power.", False),
        ("It seems to be weired.", False),
        ("He appears to have no reservation.", False),
        # complex sentences
        ("I am a student and studying informatics.", False),
        ("I think that I can win,", False),
        # question
        ("Do you know him?", False),
        ("Do you know him", False),
        ("Will you go there", False),
        ("What is it", False),
        ("Who plays tennis", False),
        ("Where do you play tennis", False),
        ("When do you play tennis", False),
        ("Why do you play tennis", False),
        ("How do you do this", False),
        # imperative
        ("Go to school.", False),
        # passive
        ("I was praised by her", False),
        ("I was not praised by her", False),
        ("I was strongly praised by her", False),
        # Future
        ("I'm going to go to school.", False),
        ("He's going to go to school.", False),
        ("We're going to go to school.", False),
        # Misc.
        ("Thank you", False),
        ("Giving", False),
        ("Made out of 100% Cotton.", False),
        ("Still here, checking this thread once in a while.", False),
        ("Of course they’ll have their down periods like anyone else", False),
        ("► ", False),
    ],
)
def test_english_extractor(text: str, is_valid: bool):
    # Set up the extractor and parser.
    extractor = EnglishExtractor("", "")  # Paths are dummy
    extractor.nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    # Parse the text.
    doc = nlp(text)
    sents = list(doc.sents)
    assert len(sents) == 1

    sent = sents[0]

    assert extractor.is_valid(sent) is is_valid
