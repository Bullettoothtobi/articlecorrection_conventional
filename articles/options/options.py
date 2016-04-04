from optparse import OptionParser


class Options:
    "Options class"

    def parse(self):
        "Parse the command line options"
        usage = 'bin/articles'
        parser = OptionParser(usage=usage)
        parser.add_option('--import',
                          nargs=1,
                          dest='importSentences',
                          default=False,
                          help='Import english sentences from file line by line')
        parser.add_option('--windowing',
                          action="store_true",
                          dest='windowing',
                          help='Windowing')
        parser.add_option('--test',
                          dest='test',
                          help='Test specific classifiers')
        parser.add_option('--on-existence',
                          action="store_true",
                          dest='existence',
                          help='Naive Bayes on classes True (article removed), False(article not removed). '
                               'Class split is 50:50')
        parser.add_option('--index',
                          action="store_true",
                          dest='index',
                          help='Index words')
        parser.add_option('--wordcount',
                          action="store_true",
                          dest='wordcount',
                          help='wordcount')
        parser.add_option('--show-words',
                          action="store_true",
                          dest='show_words',
                          help='show first 5000 words')
        return parser.parse_args();
