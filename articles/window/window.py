class Window:
    """One window"""

    def __init__(self, sequence, articles, database):
        self.sequence = sequence
        self.articles = articles
        self.db = database

    def get_articles(self):
        if self.articles.count() == 0:
            return False
        else:
            return self.articles

    def has_articles(self):
        return self.articles.count() > 0

    def get_sequence(self):
        return self.sequence

    def set_sequence(self, sequence, articles):
        self.sequence = sequence
        self.articles = articles


    # def printable(self):
    #     sequence = ""
    #     for word_id in self.sequence:
    #         word = self.db["words"].find_one(
    #                 {
    #                     "index": word_id
    #                 }
    #         )["word"]
    #         if sequence != "":
    #             sequence += ", "
    #         sequence += word
    #     articles = ""
    #     for word_id in self.articles:
    #         word = self.db["words"].find_one(
    #                 {
    #                     "index": word_id
    #                 }
    #         )["word"]
    #         if articles != "":
    #             articles += ", "
    #         articles += word
    #     return sequence + " | " + articles
