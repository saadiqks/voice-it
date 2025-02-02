from newspaper import Article

# URL of the article you want to scrape
url = "https://wesmoore.com/issues/climate/"
url = "a.b"

# Create an Article object with the given URL and language (e.g., 'en' for English)
toi_article = Article(url, language="en")

# To download the article
toi_article.download()

# To parse the article (i.e., extract the content)
toi_article.parse()

# To extract the article's title
print("Article's Title:")
print(toi_article.title)
print("\n")

# To extract the article's full text
print("Article's Text:")
print(toi_article.text)
print("\n")
