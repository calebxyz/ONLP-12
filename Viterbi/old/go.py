from sys import argv

from verification.exercise_data_building.pos import build_POS_tagging_data
from verification.model_driver_12 import model_driver_12
import solution12
import solution12_Trigrams
import solution12_trigrams_2ed
import solution12_trigrams_3rd

import itertools
N = 3
tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
#n_tag_set = list(itertools.product(tag_set, repeat=N))
n_tag_set=[]
for i in range(1, N):
    n_tag_set = list(("SOS",)*i + x for x in itertools.product(tag_set, repeat=max(0, N - i)))
#
# all_tag_set = [("SOS",)]
# for size in range(1, N + 1):
#     all_tag_set = list(itertools.product(tag_set, repeat=size))
#     for i in range(1, size):
#         all_tag_set = list(("SOS",) * i + x for x in itertools.product(tag_set, repeat=max(0, size - i)))
#
#
# start_tags = list(("SOS",)*(N-1) + (t,) for t in tag_set)
#

a = (1,2,3)
b = a[:-1]
c = (a[-1],)
def goall():
    go12()

def go12():
    # model_driver_12(
    #     solution12_Trigrams.Submission,
    #     build_POS_tagging_data(
    #         source_treebank_name="UD_English-EWT",
    #         git_hash="7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
    #     passes=3)

    model_driver_12(
        solution12_trigrams_2ed.Submission,
        build_POS_tagging_data(
            source_treebank_name="UD_English-EWT",
            git_hash="7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
        passes=3)

    model_driver_12(
        solution12_trigrams_3rd.Submission,
        build_POS_tagging_data(
            source_treebank_name="UD_English-EWT",
            git_hash="7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
        passes=3)
    #
    # model_driver_12(
    #     solution12.Submission,
    #     build_POS_tagging_data(
    #         source_treebank_name = "UD_English-EWT",
    #         git_hash = "7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
    #     passes = 3)

def print_usage():
    print()
    print("Usage:")
    print('python go.py')
    print()

if len(argv) > 1:
    print_usage()
else:
    go12()
        