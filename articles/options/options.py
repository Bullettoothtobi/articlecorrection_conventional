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
        parser.add_option('--confusion',
                          dest='confusion',
                          action="store_true",
                          help='Show confusion matrix.')
        parser.add_option('--on-pos',
                          dest='on_pos',
                          action="store_true",
                          help='Test on part of speech')
        parser.add_option('--app-pos',
                          dest='app_pos',
                          action="store_true",
                          help='Test on sentence and appended part of speech')
        parser.add_option('--app-phoneme',
                          dest='app_phoneme',
                          action="store_true",
                          help='Test on sentence and appended first phoneme')
        parser.add_option('--report',
                          dest='report',
                          action="store_true",
                          help='Show full classification report.')
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
