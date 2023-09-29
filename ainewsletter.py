from newsapi import NewsApiClient
from langchain import PromptTemplate
from datetime import datetime, timedelta
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import ast
from bs4 import BeautifulSoup
from urllib.request import urlopen
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


class printHTML:
        def generate_html_table(articles):
            table = '<table border="1">\n'

            # Add table headers
            table += "  <thead>\n    <tr>\n"
            for header in ["Title", "URL", "Summary"]:
                table += f"      <th>{header}</th>\n"
            table += "    </tr>\n  </thead>\n"

            # Add table rows
            table += "  <tbody>\n"
            for article in articles:
                table += "    <tr>\n"
                table += f"      <td><a href='{article['url']}'>{article['title']}</a></td>\n"
                #table += f"      <td>{article['url']}</td>\n"
                table += f"      <td>{article['summary']}</td>\n"
                table += "    </tr>\n"
            table += "  </tbody>\n"

            table += "</table>"
            return table
        
        

class getText:
       def get_textfromURL(url):
        page = urlopen(url)
        html = page.read().decode('utf-8')
        cleaned_content = html.replace(u'\u2013', '-')
        soup = BeautifulSoup(cleaned_content, 'html.parser')
        # print the text retrieved
        #print(soup.get_text())
        return soup.get_text()

class getSummary:
    def get_summary(text):

         combine_template = """
                        Your job is to produce a concise summary of a news article about Machine Learning
                        Provide the important points and explain why they matter
                        The summary is for a blog so it has to be exciting to read

                        Write a concise summary of the following:

                        {text}

                        CONCISE SUMMARY:
                        """
         combine_prompt = PromptTemplate(
            template=combine_template, 
            input_variables=['text']
            )
         char_text_splitter = CharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 0
        )
         # we need to cast the data as a Document
         doc = [Document(page_content=text)]
         #print(f"doc is here: {doc}")
         # we split the data into multiple chunks
         docs = char_text_splitter.split_documents(doc)
         # we load the chain with ChatGPT
         llm = ChatOpenAI(model= "gpt-3.5-turbo", 
                         openai_api_key="{Provide you key here}")
         chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     combine_prompt=combine_prompt)
         summary = chain.run(docs)
         print(f"summary is {summary}")
         return summary

class extractNews:
    def getNews(self):
        NEWS_API_KEY = "{provide your key here}"

        newsapi = NewsApiClient(NEWS_API_KEY)

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # print(f"Today's date: {today}")
        # print(f"Yesterday's date: {yesterday}")

        everything = newsapi.get_everything(
            q='artificial intelligence',
            from_param= today, #"2023-09-27",
            to= yesterday,   #"2023-09-26",
            sort_by='relevancy',
            language='en'
        )
        #print(type(everything))
        #return list(everything.items()) #['articles'][0]
    
        # Select top 10 news articles
        top_news_template = """
                    extract from the following list the 10 most important news about machine learning.
                    Return the answer cleaning up the text with no special characters etc. and no special encodings so that I can load in python lists.

                    For example: ['text 1', 'text 2', 'text 3']

                    {list}
                """
        TOP_NEWS_PROMPT = PromptTemplate(
            input_variables= ['list'],
            template = top_news_template,
        )

        # Get the titles
        title_list = '\n'.join([article['title'] for article in everything['articles']])
        #print(title_list)

        # Summarize the top 10 articles that were selected
        llm = ChatOpenAI(model= "gpt-3.5-turbo",
                         openai_api_key="{provide your key here}")
        chain = LLMChain(
            llm = llm,
            prompt = TOP_NEWS_PROMPT
        )

        # now let's convert the result into a Python list
        # we send the prompt to ChatGPT
        top_str = chain.run(title_list)

        # convert the result to python list
        top_list = ast.literal_eval(top_str)
        #print(top_list)

        # new let's match those titles with the related URLs
        top_news = [
            {
                'title': a['title'],
                'url': a['url']
            }
            for a in everything['articles']
            if a['title'] in top_list
        ]
        #print(top_news)
        # Get the first URL for testing
        # url = top_news[0]['url']
        # use beautifulsoup to retreive html from all urls
        # print (url)
        for news_dict in top_news:
            try:
                text = getText.get_textfromURL(news_dict['url'])
                #news_dict['text'] = text
                # print(f"news_dict[text]: {news_dict['text']}")
                summary = getSummary.get_summary(text)
                news_dict['summary'] = summary
                #print(f"news_dict[summary]: {news_dict['summary']}\n")
                #exit
            except Exception as e:
                #news_dict['text'] = None
                news_dict['summary'] = None
                #print(f"Exception occured. {e.with_traceback}")
                continue

        print(top_news)
        html_table = printHTML.generate_html_table(top_news)
        print(html_table)


       
    

if __name__ == '__main__':
    retrieveNews = extractNews()
    list_of_articles = retrieveNews.getNews()
    #print(type(list_of_articles))