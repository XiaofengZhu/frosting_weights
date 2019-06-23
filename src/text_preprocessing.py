import re
from bs4 import BeautifulSoup


class PreprocessMethods:
	def strip_html_urls(self, text):
		text = str(text)
		text = BeautifulSoup(text, 'lxml').text
		text = re.sub(r'http\S+', r'<URL>', text)
		text = re.sub('\n', ' ', text)
		text = text.lower()
		return text


if __name__ == '__main__':
	pm = PreprocessMethods()
	text = pm.strip_html_urls('https://THIS IS A TEST')
	print(text)