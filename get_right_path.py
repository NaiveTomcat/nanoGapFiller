import sys
import getopt
import itertools

import fastg_file
from Bio import SeqIO

def parse_node_long_name(long_name):
    short_name, _ = fastg_file.read_long_name(long_name)
    uid, length, _, is_reverse = fastg_file.read_short_name(short_name)
    uid += ('r' if is_reverse else '')
    return uid, length

class Alignment:

    VALID_THRESHOLD = 0.95

    def __init__(self, alignment_line):
        self.line = alignment_line
        tokens = alignment_line.rstrip().split('\t')
        self.query_id = tokens[0]
        self.subject_id = tokens[1]
        self.query_node_id, query_node_len = parse_node_long_name(
            self.query_id
        )
        identity = float(tokens[2]) / 100
        alignment_length = int(tokens[3])
        # num_error = int(tokens[4])
        self.gap_open = int(tokens[5])
        q_start, q_end, s_start, s_end = map(int, tokens[6:10])
        self.start_cut = q_start - 1
        self.end_cut = query_node_len - q_end
        self.num_delete = alignment_length - (abs(q_start - q_end) + 1)
        self.num_insert = alignment_length - (abs(s_start - s_end) + 1)
        self.e_value = float(tokens[10])
        self.bit_score = float(tokens[11])
        self.start = s_start
        self.end = s_end
        self.left = min(s_start, s_end)
        self.right = max(s_start, s_end)
        self.identity = alignment_length * identity / query_node_len
        self.is_valid = True if\
            self.identity > Alignment.VALID_THRESHOLD else False
        self.is_forward = s_end > s_start
        self.children = []

    def add_child(self, alignemnt):
        self.children.append(alignemnt)

    def adjacent_before(self, alignment, overlap):
        """This mean self is adjacent to `alignment` and 
        `self` is on the upper stream of `alignment`."""
        min_insert = min(self.num_insert, alignment.num_insert)
        min_delete = min(self.num_delete, alignment.num_delete)
        # To be True, two alignment must have the same direction.
        is_same_direction = (self.is_forward == alignment.is_forward)
        shift = overlap - 1
        if not is_same_direction:
            return False
        if self.is_forward:
            real_self_end = self.end + self.end_cut
            result_other_start = alignment.start - alignment.start_cut
            return result_other_start  - min_insert <= \
                real_self_end - shift <= result_other_start + min_delete
        else:
            real_self_end = self.end - self.end_cut
            real_other_start = alignment.start + alignment.start_cut
            return real_other_start - min_delete <= \
                real_self_end + shift <= real_other_start + min_insert

    @classmethod
    def index(cls, alignments, key):
        if key == 'node id':
            key_attr = 'query_node_id'
        elif key == 'start position':
            key_attr = 'start'
        else:
            raise ValueError('Key not supported.')
        index = {}
        for alignemnt in alignments:
            if getattr(alignemnt, key_attr) in index:
                index[getattr(alignemnt, key_attr)].append(alignemnt)
            else:
                index[getattr(alignemnt, key_attr)] = [alignemnt]
        return index

    @classmethod
    def add_connection(cls, alignments, nodes):
        """Only for those forward alignments"""
        alignments = list(filter(lambda x: x.is_forward, alignments))
        # Sort method.
        alignments.sort(key=lambda x: x.start)
        for i in range(len(alignments)):
            for child_node, overlap in\
                    nodes[alignments[i].query_node_id].children:
                stop_position = alignments[i].end - (overlap - 1) + 50
                j = i + 1
                while j < len(alignments) and alignments[j].start < stop_position:
                    if alignments[j].query_node_id == child_node.uid and \
                            alignments[i].adjacent_before(alignments[j], overlap):
                        alignments[i].add_child(alignments[j])
                    j += 1

def read_file(file_name):
    alignemnts = []
    with open(file_name) as fin:
        for line in filter(lambda x: not x.startswith('#'), fin):
            # Parse a line.
            alignemnt = Alignment(line)
            alignemnts.append(alignemnt)
    return alignemnts

def is_adjacent(node_a, node_b, node_id2alignments, overlap):
    for alignment_a, alignment_b in itertools.product(
            node_id2alignments[node_a], node_id2alignments[node_b]):
        if alignment_a.adjacent_before(alignment_b, overlap):
            return True
    return False

def write_file(output_file, alignments):
    fout = open(output_file, 'w')
    for alignment in alignments:
        fout.write(alignment.line)
        for child in alignment.children:
            fout.write('\t')
            fout.write(child.line)
        fout.write('\n')
    fout.close()

def printHelpMessage():
    body = '[-h] <-l overlap len> <fastg file> <blast result> <output>'
    print('python3 {} {}'.format(__file__, body))

def main():
    fastg_file_name = ''
    blast_result_file = ''
    output_file = ''
    overlap_len = None
    options, args = getopt.getopt(sys.argv[1:], 'hl:')
    for option, value in options:
        if option == '-l':
            overlap_len = int(value)
        elif option == '-h':
            printHelpMessage()
            sys.exit()
        else:
            printHelpMessage()
            sys.exit()
    fastg_file_name, blast_result_file, output_file = args

    nodes = fastg_file.build_assembly_graph(fastg_file_name, overlap_len)
    alignments = list(filter(lambda x: x.is_valid and x.is_forward,
        read_file(blast_result_file)))
    Alignment.add_connection(alignments, nodes)
    alignments.sort(key=lambda x: x.start)
    write_file(output_file, alignments)

if __name__ == '__main__':
    main()