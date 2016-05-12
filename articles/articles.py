import sys

from options import options
from database import database

if __name__ == '__main__':
    options = options.Options()
    opts, args = options.parse()

    db = database.Database()

    if opts.importSentences:
        db.import_sentences(opts.importSentences)

    if opts.windowing:
        db.windowing()

    if opts.index:
        db.create_wordindexes()

    if opts.wordcount:
        db.get_wordcount()

    if opts.show_words:
        db.show_words()

    if opts.test:
        db.test(classifier_name=opts.test, confusion=opts.confusion, report=opts.report,
                on_pos=opts.on_pos, app_pos=opts.app_pos, app_phoneme=opts.app_phoneme,
                no_source_word=opts.no_source_word)
