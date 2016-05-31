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
        parser.add_option('--app-position',
                          dest='app_position',
                          action="store_true",
                          help='Test on sentence and appended position of the article')
        parser.add_option('--app-phoneme',
                          dest='app_phoneme',
                          action="store_true",
                          help='Test on sentence and appended first phoneme')
        parser.add_option('--no-source-word',
                          dest='no_source_word',
                          action="store_true",
                          help='Leave out the source word when creating the vector.')
        parser.add_option('--report',
                          dest='report',
                          action="store_true",
                          help='Show full classification report.')
        parser.add_option('--on-existence',
                          action="store_true",
                          dest='on_existence',
                          help='Test only whether an article is expected or not.')
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
        parser.add_option('--ngrams',
                          nargs=1,
                          dest='ngrams',
                          default=1,
                          help='use n-grams instead of single words')
        return parser.parse_args();
