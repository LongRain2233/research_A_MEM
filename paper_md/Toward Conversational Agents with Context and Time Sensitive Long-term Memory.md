# Toward Conversational Agents with Context and Time Sensitive Long-term Memory

Nick Alonso

Zyphra

nick@zyphra.com

Tomás Figliolia

Zyphra

tom@zyphra.com

Anthony Ndirango

Zyphra

anthony@zyphra.com

Beren Millidge

Zyphra

beren@zyphra.com

# Abstract

There has recently been growing interest in conversational agents with long-term memory which has led to the rapid development of language models that use retrieval-augmented generation (RAG). Until recently, most work on RAG has focused on information retrieval from large databases of texts, like Wikipedia, rather than information from long-form conversations. In this paper, we argue that effective retrieval from long-form conversational data faces two unique problems compared to static database retrieval: 1) time/event-based queries, which requires the model to retrieve information about previous conversations based on time or the order of a conversational event (e.g., the third conversation on Tuesday), and 2) ambiguous queries that require surrounding conversational context to understand. To better develop RAG-based agents that can deal with these challenges, we generate a new dataset of ambiguous and time-based questions that build upon a recent dataset of long-form, simulated conversations, and demonstrate that standard RAG based approaches handle such questions poorly. We then develop a novel retrieval model which combines chained-of-table search methods, standard vector-database retrieval, and a prompting method to disambiguate queries, and demonstrate that this approach substantially improves over current methods at solving these tasks. We believe that this new dataset and more advanced RAG agent can act as a key benchmark and stepping stone towards effective memory augmented conversational agents that can be used in a wide variety of AI applications.1

# 1 Introduction

Conversational agents, such as chatbots, personal assistants, and language interfaced operating systems, are currently seeing rapid development and interest both in academia and industry (e.g., [11, 19, 32, 14, 17, 29]). One specific area of interest is in conversational agents that utilize retrievalaugmented generation (RAG) to imbue these agents with long-term memory. However, popular QA benchmarks that typically test RAG systems, focus primarily on information retrieval from a static database of texts, such as Wikipedia (e.g., [13, 30]. However, the increasing importance of conversational agents raises the question of how to address the unique challenges RAG models face in conversational contexts that they do not in offline, database retrieval contexts. In this paper, we focus on what we see as two crucial challenges conversational agents face that are not tested in most standard database retrieval benchmarks:

1. Conversational Meta-Data Based Queries. In conversational contexts, a common sort of query refers to meta-data (e.g. time, date, or speaker) associated with previous conversations. For example, one could plausibly ask "what were we discussing yesterday morning, again?", "what was that idea we were working on last time?", or "summarize what Jason talked about in our meeting from January 6th.". These questions are not specifying what was talked about, but are instead asking the model to specify what was talked about given some meta-data (such as time) associated with a conversational event. Such questions point to a class of common questions an conversational agent may face, which cannot be answered without some ability to retrieve information about previous conversations based on conversational meta-data, rather than semantic retrieval alone.

2. Ambiguous Questions. In conversation, it is normal to speak with pronouns (he, she, it, they, etc.) and demonstratives (’this’, ’that’, etc.), which are ambiguous without an understanding of preceding conversational context. Although understanding this context is trivial for generation by LLMs, such statements will fool naive RAG systems as we discuss below.

There is currently a lack of good benchmarks and models that explicitly and directly address these challenges for retrieval systems for conversational agents. Although, there exists a QA benchmark that tests ambiguous questions [8, 9], it does not test conversational agents that are retrieving from records of chat history and does not test ambiguous queries that refer to meta-data rather than text content. Further, recent work has created benchmarks which test long-term memory in conversational agents (e.g., [11, 19, 4]) that do not directly, or deeply test meta-data retrieval or ambiguous questions. With respect to models, although IR models have long existed for searching through databases with meta-data, such as IR models for tabular data, such models are not standard in conversational agents that use RAG (e.g., see [11, 19, 32, 14, 4, 17]), which primarily utilize semantic, vector database search.

There is, therefore, a need for datasets and benchmarks that directly test a conversational agent’s ability to recall information based on queries which are ambiguous and which refer to conversational meta-data. In this paper, we attempt to make progress in this direction through two main contributions. First, we construct a dataset and benchmark for conversational agents that directly tests these two capabilities. We build upon an existing dataset of long-form dialogues to generate a dataset of questions that 1) refer to conversational meta-data, 2) ambiguous questions that refer to conversational meta-data, and 3) questions that refer to a combination of meta-data and conversational content. Second, we develop a novel retrieval model that combines standard vector database search with a tabular search method known as chain-of-tables [28] for retrieving from chat logs using meta-data based queries, and we improve the performance of this model through the use of a classifier that decides whether the query is referencing meta-data, content, or both. We find that this system significantly outperforms existing retrieval systems in the literature, especially w.r.t. the common problems we have identified. Ultimately, we believe that our work can act as a stepping stone toward more intelligent conversational agents with long-term memory.

# 2 Related Works

Long-Context LLMs Significant recent attention has also been paid to ’needle in the haystack’ tasks for LLMs with long contexts, where the task requires reporting information stored in a small portion of text present in long, usually $> 1 0 0 { , } 0 0 0$ , context of an LLM (e.g., [10, 5, 1, 3, 18]). Long-context models often have difficulty finding and attending to relevant tokens while ignoring all the other information stored in context. We are similarly interested in how a model may retrieve information contained in a small portion of text that exists in a long stream of input text. However, contrary to this line of work, we are 1) interested in computationally cheap, affordable ways of performing this task that does not require long context models, and 2) interested in search for a small chunk of text based on its meta-data (e.g., date or time) rather than on content contained in its text.

Retrieval Augmented Generation RAG models [16] work by storing a database of text chunks, each of which is associated with a semantic embedding vector, which is then used as a key during vector database retrieval. Retrieved text can be injected directly into the context or injected at hidden layers of LLMs [2] RAG models often show improved performance, both in terms of perplexity and QA accuracy, over LLMs that do not utilize RAG but have the same or more overall size (e.g.,

![](images/f2ea60fff8f142f981e4d0014b8ace6498e96489a62abeaaa597681e9f0b2fe3.jpg)  
Figure 1: Depiction of our combined tabular and semantic vector-search method.

[16, 2]. RAG models are often used on QA benchmarks that use a standard database of texts like Wikipedia (e.g., [13, 30]. More recently, RAG applied to streams of conversational responses has been investigated, which we now discuss.

Long-term Dialogue. There has been a growing interest in testing information retrieval (IR) in conversational agents (e.g., [11, 19, 32, 14, 17, 29]). Unlike benchmarks that test IR on static datasets, such as Wikipedia (e.g., [13, 30]), these benchmarks seek to test a conversational agent’s ability to recall and report information that may have been communicated by the user far in the past of the conversation history (i.e. out of immediate context of the model). Until very recently, however most datasets of conversation logs were relatively short, involving only a few sessions/conversations between an agent and a single user (e.g., [11, 32, 14, 17, 29]). More recently, there have been efforts to create benchmarks that involve much longer dialogues to better test long-term memory abilities vs within context retrieval. The company GoodAI recently released a series of benchmarks that test a conversational agent’s ability to retrieve information that may have been communicated tens or hundreds of thousands of tokens in the past [4]. This test works by interleaving simulated small talk and filler text with questions from the user. The recently developed LoCoMO dataset [19] is a high-quality dataset of long-form conversations between two simulated, GPT4-based agents that involve many sessions worth of conversation. The LoCoMo benchmark uses questions that involve summarization of past portions of conversation, one-hop and multi-hop question answering, and temporal reasoning. Both the GoodAI and LoCoMo benchmarks test IR in conversational agents in many useful ways, but do not directly test or focus on ambiguous question answering or meta-data-based IR.

Temporal Reasoning in LLMs. Although, the LoCoMo dataset [19] tests temporal reasoning, this reasoning only requires the retrieval of items using content-based retrieval, e.g., the question ’how long did it take for Greg to write his novel?’, only requires a semantic retrieval system to retrieve statements about novel writing along with their associated timestamps. Other tests of temporal reasoning in LLMs, such as the benchmark by [27], focus on LLMs ability to reason about the frequency, ordering, duration, causality, temporal differences, and other related temporal properties, e.g., multiple choice questions like "Form the following events in chronological order." or "Which of the following events is the longest?". These questions do not specifically require retrieval items from a memory bank based on their temporal properties (e.g., when they occurred) since the questions are usually one-off questions that provide the relevant events names/descriptions within the question.

Ambiguous and Conversational Querying. Common QA benchmarks for language models, like Natural questions [13] or HotPotQA [30], use questions with subjects and objects that are explicitly named, e.g., "what color was John Wilkes Booth’s hair?" [13], or explicitly described "What was the former band of the member of Mother Love Bone who died just before the release of “Apple”?" [30]. The same is true of the recent conversational memory benchmarks from GoodAI and LoCoMo. As far as we can find, the "The Conversational Assistance Track Overview" (CAsT) benchmark [8, 9] is the only benchmark for testing IR queries generated in a conversational context that are ambiguous. All versions of the test involve retrieving from database of passages from Wikipedia, news articles, and other similar databases. The CAsT benchmark does not focus on retrieving from tabular data of conversation logs between two agents and thus does not suit our needs, but we take inspiration from this dataset in generating our own dataset, as we describe below.

Table-based RAG. Popular table-base retrieval datasets, like TabFact [6] and WikiTQ [23], involve the retrieval of information from a dataset of tabular data containing general factual information. These datasets do not test ambiguous, conversational queries and do not focus on conversation logs as we desire to do here. Current SOTA RAG systems for tabular databases write code to query and manipulate the table. For example, text-to-SQL models have shown favorable results in querying and retrieving from tabular databases, e.g., [7, 31]. More recently, a chain-of-table (CoTable) model which chains together a series of python-like function calls shows superior performance [28]. We take inspiration from this approach in designing a model below.

Chat-bots with Long-term Memory. There has been a recent proliferation of chatbots with RAG both in research settings and in industry. Many open-source chatbots with memory use relatively simple semantic retrieval techniques (e.g., [11, 19, 32, 14, 17, 29]). A growing number are using more advanced query and memory writing techniques (e.g., [17, 19, 4]). However, as far as we can find, these systems still do pure semantic retrieval or at least do not directly test the ability of the model to retrieve information based on tabular meta-data, such as queries about the time of a conversation, or the ability of the model to handle ambiguous queries. Without standard benchmarks for conversational agents which test these capabilities, it is hard to compare the performance of the various conversational agents that are being rapidly produced. We hope our work will provide a useful foundation toward resolving this issue.

# 3 Dataset

The LoCoMo Dataset. The LoCoMo dataset [19] consists of 35 high-quality dialogues between two agents simulated based on the reflect and respond architecture [22] using GPT4 augmented with persona prompts, a short-term memory that summarizes recent statements, and long-term memory that consists of generated observations/facts about the personas life. The generated conversations are edited by hand to ensure the conversations are consistent, non-contradictory, etc. Each conversation set consists of about 19 sessions/conversations on average, with an average of 9,209 tokens per dialogue. The LoCoMo benchmark consists of event summarization and question answering tasks, which include single-hop and multi-hop content-based questions, general knowledge questions, temporal reasoning, and an adversarial task that attempts to trick the agent into giving the wrong answer. At the time of writing, the dataset of the dialogue has been released publicly, while the questions have not been released. Nonetheless, it is clear that none of these tests focus specifically on ambiguous queries. Further, although the benchmark investigates QA tasks, the questions involved are, as far as we can tell, not designed to specifically test the ability of a model to recall portions of conversation based on meta-data retrieval, e.g., based on when the conversation occurred. The temporal reasoning task may be the closest to what we are looking for but still does not require meta-data based retrieval. Example questions from this task, such as ’How long did it take for X to finish writing the book?’ [19], only require content based retrieval of conversation snippets related to book writing along with their time-stamps, which the LLM can then reason about.

Modification to Locomo. Despite the questions not yet being released, the released dialogues from LoCoMo can be used to generate new types of questions. Before generating new questions, we first made three adjustments to the original dataset. First, some of the original 35 dialogues were short, e.g., $< 4 0 0 0$ tokens, which can fit directly into the context of many modern LLMs and is hence unsuitable for testing long term memory. We used only the 12 longest dialogues from the dataset. Second, we added a new conversation/session, of about 4000 tokens, to the end of each of the remaining dialogues to act as padding. We do not ask the model about this extra conversation. This ensures that the short (4k) context LLMs we are interested must use long-term memory unit that removes irrelevant recent statements from the context and replaces them with relevant statements from long-term memory. Third, the original dataset had a time and date stamp for each session/conversation. We added additional time and date stamps to each individual response where we simulate the elapsed time of a response by assuming agents generate language at a rate equal to the average number of words spoken for humans [25]. This augmentation allows us to test more advanced explicit time-based querying of the dataset.

Test Set-Up Instead of storing the desired output text/answer for each question, we store a list of the relevant responses that should be retrieved from the chat log by the IR system. This approach is similar to other IR benchmarks (e.g., [9, 8]) which seek to dissociate the performance of the IR/memory module from the language model that generates the answer. We believe this is a better

![](images/56434f691a4a213b66c662ecb7ec567d62fceb03ca855f502d603ab3a1c71d71.jpg)  
Figure 2: Examples of queries for various types in our dataset.

way to isolate the performance of an IR model from that of the LLM. To measure performance, we compute the recall and F2 scores of the recalled response numbers. Recall is the proportion of relevant responses that are retrieved. F2 is the weighted harmonic mean of recall and precision, where precision is the proportion of recalled items that are relevant (see appendix A.2 for details). We use F2 instead of F1, since F2 places more emphasis on recall than precision, which better captures the desirable qualities of an IR system used with an LLM. LLMs must have relevant information in context to correctly answer a question (therefore recall of the IR system is essential). Further, it has been repeatedly found (e.g., [19, 4, 15]) that LLMs can ignore some but, not all, irrelevant text in context, so precision is important but less essential than recall.

Our tests assume that the agent stores the chat-log in its entirety. We believe this is feasible since chat logs, even those that may occur over the course of years, are small relative to the massive databases of texts, like wikipedia that are more commonly used in information retrieval. Nonetheless, this test can be easily adapted to slightly different forms of datasets, e.g., those that store a summary of each session instead of all responses from each session.

Time-based Queries. We write by hand a set of 11 different types/templates of single-hop, timebased queries that may occur in realistic conversational settings. For each question type, we use the template to automatically generate a large set of possible forms that the question can take for a given conversational set, e.g. for the question template "What did we discuss in session N?", there are N questions generated of this type per conversation set, where N is the number of sessions/conversations in the set. We then generate multiple versions of the question each with significantly different wording using several different hand crafted templates of the same question and by swapping out different spellings of numbers (e.g., writing the question both with "5th" and "fifth") for increased diversity of samples. For more detail on time based query types see appendix A.3.

Ambiguous Time-based Queries. Next, we create ambiguous versions of the time based queries described above. We do so by creating by hand three initial conversation templates, which begin by making a comment about a conversation at a specific point in time, then proceed to talk about the conversation using demonstratives and pronouns (e.g., "it" or "that"), then finish by asking the model to summarize the conversation using a pronoun or demonstrative, such as "When did we discuss that earlier?". Multiple different wordings for the first response in the conversation are also created to generate and increase diversity of conversational templates, as well as generated conversation with different wording of numbers, times, and dates (e.g., "fifth" and "5th"), resulting an at least 6-12 different conversations per question.

Time+Content Queries. To better test an IR system’s ability to answer more complex queries we generate multi-hop questions that require both content and meta-data based retrieval. We use GPT4 to generate the majority of the questions, by prompting it with several hand-written examples. We then edit by hand the generated questions to fix inconsistencies and ensure questions and associated response number of accurate. These questions refer to three properties of a response: 1) the speaker, 2) the date or session number, and 3) the general (non-specific) content/topic of the response, which

provides some information on the topic of the response but not enough for highly accurate semantic retrieval. An example question is "What video game did Jolene mention playing with her partner on January 27th, 2023?". To get a high F2 score, it is not enough to retrieve all response related to video games, since there are dozens in the conversation logs. Nor is it enough just to retrieve all responses from January 27th or all of Jolenes responses since this will also retrieve many irrelevant responses. To maximize precision, the model must isolate the relevant response using all three features.

# 4 Model

We test and describe several simple baselines in the next section. Here we describe the more advanced information retrieval system we have designed to deal with ambiguous and meta-data-based queries in our novel dataset and benchmark tasks.

Tabular Chat Database. Each response consists of text as well as meta-data, such as speaker name, date, time, session number, etc. This naturally lends itself to tabular database representation, where each column in the table consists of data of a particular type (e.g., time) and each row consists of information related to a specific response. We combine a this table with a vector database by creating a ’Content’ column that stores the index of the associated semantic vector in the vector database of responses. This table can then be queried for content by retrieval response from the top-k found through semantic retrieval, along with any meta-data based queries. We describe how we combine multiple queries.

Classifying Query Type. Some questions may require retrieving responses based on semantic retrieval, where a semantic embedder is used to create and retrieve semantically related responses from a vector database. Other questions may just require querying the table for responses with relevant meta-data. Others may require both types of query simultaneously. We use an LLM to classify whether a given query requires meta-data retrieval (by outputting yes or no) and/or semantic retrieval (yes or no). This LLM call uses few shot prompting with examples. We show empirically below that such classification can be highly accurate, and improves performance over methods which require the model to both create complex database queries and decide which type of queries to perform in the same response.

Chain-of-Tables for Meta-Data Retrieval. Tabular databases can be queried using standard specialized programming languages such as SQL or data-frame based python languages (e.g., pandas). One recent high performing method for querying tabular data, which outperformed SoTA text2SQL approaches, is the chain-of-table algorithm [28]. This algorithm creates a small library of functions that operates on and retrieves elements from the table. An LLM is used to call a chain of these functions in a sequence to perform advanced multi-hop queries on the table. We adapt chain of tables for meta-data queries on conversational logs. We use two functions to retrieve subsets of rows from the table:

• f_value(column_name, [value1, value2,...])   
• f_between(column_name, [value1, value2])

The f_value function retrieves all rows that match at least one of the listed values in the given column. The f_between function retrieves all rows in between value1 and value2. Although these functions are simple we find they can be used to accurately complete, in principle, all of the questions in our test set. We adapt the prompts from the original chain of table method to this custom function library. We use an LLM to write the chain using separate prompts to write 1) the function name, 2) the first argument (column name) and 3) the second argument (values) of the function. These prompts can be found in appendix A.7.

Combining Meta-Data and Semantic Retrieval. We combine semantic and meta-data retrieval in the following way. If a query is classified as requiring both meta-data and content based retrieval, we first do a meta-data based retrieval using the chain-of-table method described above. This outputs a subset of rows from the original table. Then we to a semantic search to retrieve the top-k related related responses from the remaining table. The intuition for ordering queries in this way is that tabular data search is a well-known and optimized problem which can be fast and deterministic while semantic search using embeddings is a stochastic and computationally expensive problem. Thus, reducing the search space by first running the cheap, deterministic queries, results in a more efficient and reliable execution.

<table><tr><td rowspan="2" colspan="2">Retrieval Method</td><td colspan="2">Time Qs</td><td colspan="2">Time+Content Qs</td><td colspan="2">Average</td></tr><tr><td>Recall</td><td>F2</td><td>Recall</td><td>F2</td><td>Recall</td><td>F2</td></tr><tr><td rowspan="3">Semantic</td><td>(k=10)</td><td>2.01</td><td>2.32</td><td>15.43</td><td>5.62</td><td>8.72</td><td>3.97</td></tr><tr><td>(k=20)</td><td>3.91</td><td>4.21</td><td>24.29</td><td>5.19</td><td>14.10</td><td>4.70</td></tr><tr><td>(k=30)</td><td>5.82</td><td>5.89</td><td>29.43</td><td>4.43</td><td>17.62</td><td>5.16</td></tr><tr><td rowspan="3">Semantic w/MetaD</td><td>(k=10)</td><td>2.51</td><td>2.90</td><td>37.83</td><td>13.68</td><td>20.17</td><td>8.29</td></tr><tr><td>(k=20)</td><td>5.02</td><td>5.43</td><td>51.26</td><td>10.85</td><td>28.14</td><td>8.14</td></tr><tr><td>(k=30)</td><td>7.47</td><td>7.55</td><td>56.40</td><td>8.43</td><td>31.93</td><td>7.99</td></tr><tr><td colspan="2">CoTable+Semantic (hMistral7b)</td><td>93.95</td><td>87.67</td><td>65.30</td><td>22.69</td><td>79.62</td><td>55.18</td></tr><tr><td colspan="2">CoTable+Semantic (GPT3.5)</td><td>90.47</td><td>78.34</td><td>90.17</td><td>32.19</td><td>90.32</td><td>55.27</td></tr></table>

Table 1: Recall and F2 scores for unambiguous time-based and time+content-based questions. All models use the original question to query the database.

Query Rewrite To deal with ambiguous queries we adapt a SoTA prompting based query rewriting method [20], which uses few-shot prompting techniques to show an LLM how to disambiguate a query given the query and preceding context. We adapt this prompt to include examples similar to our dataset, and to make sure the model does not significantly alter the query if unambiguous (see appendix A.7). Although other methods exist that directly map ambiguous queries to embedding vectors (e.g., [21]), these require retraining or fine-tuning which may not be possible when accessing LLMs remotely via API.

In sum, when a query is given to our full model, it first disambiguates the query using an LLM for query rewriting. Then the query is classified as either needing only meta-data based retrieval, only semantic retrieval, or both. Finally, chain of tables and/or semantic retrieval is used to return the relevant responses and their meta-data. Pseudo-code for this algorithm is shown in appendix A.6.

# 5 Experiments

We compare our model to two baselines. First is a simple semantic retrieval system that stores a vector database of semantic vectors each representing the text of one response (semantic). Second, we test a semantic retrieval system that using a more advanced writing technique, where the meta-data is explicitly written in text then concatenated to the response text before, the theory being that this could allow pure semantic search to perform meta-data lookup by similarity matching. This combined text chunk is used to generate the semantic embedding for each response (Semantic w/ Meta Data).

We test our combined CoTable+Semantic IR system using two different LLMs with short 4k contexts for the rewrite and CoTable writing. First, we test the model with an open-source fine-tuned version of Mistral 7b [12] called OpenHermes (hMistral-7b) [26], which we found performed noticeably better at writing python-like function calls during chain of table than the base Mistral model. Second, we test our model with GPT-3.5-turbo, a large model with 4k context that is relatively cheap in cost compared to its GPT4 counterparts. All of the CoTable $^ +$ Semantic models retrieve $_ { \mathrm { k = 1 0 } }$ values when using the semantic retrieval. All models use the same semantic embedder, the multi-qa-mpnet-base-dot-v1 model [24], which is a high-performing, open source semantic embedder trained specifically for QA.

Time based Queries. Results for time based query test (unambiguous questions) can be found in table 1. The semantic retrieval model failed the time-based queries completely, which is to be expected since it uses embedding vectors that carry no information about the meta-data of the response. The semantic search with meta-data imbued embeddings performed slightly better but still performed very poorly overall, getting recall and F2 scores less than 10. Our CoTable+Semantic retrieval method performed well getting 90 recall accuracy for both LLMs and around 75-78 F2. Interestingly, GPT3.5 performed slightly worse than hMistral-7b. We find this may be due to the meta-semantic classifier being slightly less accurate with GPT-3.5 for these questions (table 4).

Time+Content Queries. Results for the test with combined time and content based queries can be found in 1. The basic semantic retrieval did better than it did on the pure time-based questions, which is to be expected because the embedding vectors now carry information relevant to the query, which

![](images/5f0d3a6ada3371ea3e6f81cda696b473f5391b0080be9a325d67e0fa709c4fac.jpg)  
Figure 3: F2 scores for each individual time-based test and the time+content based test. All models use $_ { \mathrm { k = 1 0 } }$ for semantic search. Error bars show std. of recall and precision across data in each test.

Table 2: Recall and F2 scores for ambiguous time-based questions. All models use $\overline { { \mathrm { k } { = } 1 0 } }$ for semantic search. Models either use the original, ambiguous question to query the database (None), the query and the 3-6 preceding responses (context), or an LLM query writer prompted to disambiguate the query (Qry Write).   

<table><tr><td rowspan="2">Retrieval Method</td><td rowspan="2">Query Type</td><td colspan="2">hMistral7b</td><td colspan="2">GPT-3.5</td></tr><tr><td>Recall</td><td>F2</td><td>Recall</td><td>F2</td></tr><tr><td rowspan="3">CoTable+Semantic</td><td>Original Qry</td><td>2.93</td><td>2.35</td><td>10.62</td><td>3.12</td></tr><tr><td>Context+Qry</td><td>73.51</td><td>61.59</td><td>77.27</td><td>65.47</td></tr><tr><td>Qry Rewrite</td><td>89.43</td><td>81.05</td><td>83.9</td><td>72.56</td></tr></table>

discusses the content of the relevant response. However, because these questions are not especially specific with respect to the content of the referenced response, and because this simple IR system is insensitive to the meta-data, it still performed poorly, getting a maximum recall around $2 8 \%$ and F2 around 5.6. The semantic IR with meta-data imbued semantic vectors performed noticeably better than basic semantic IR getting recall closer to $5 6 \%$ and maximum F2 around 13. However, the CoTable+Semantic models performed much better, with the GPT3.5 model doing the best. The hMistral model obtains around $6 5 \%$ recall accuracy and an F2 of 22.69. The GPT-3.5 version obtains $9 0 . 1 7 \%$ recall and around 32 F2.

Table 3: Testing the models with and without a meta-semantic classification step. Recall and F2 scores for unambiguous time-based and time+content-based questions.   

<table><tr><td rowspan="2">Retrieval Method</td><td colspan="2">Time Qs</td><td colspan="2">Time+Content Qs</td><td colspan="2">Average</td></tr><tr><td>Recall</td><td>F2</td><td>Recall</td><td>F2</td><td>Recall</td><td>F2</td></tr><tr><td>CoTable+Semantic (hMistral7b)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/ meta-semantic classification</td><td>93.95</td><td>87.67</td><td>65.30</td><td>22.69</td><td>79.62</td><td>55.18</td></tr><tr><td>w/o meta-semantic classification</td><td>42.26</td><td>33.11</td><td>69.07</td><td>28.29</td><td>18.82</td><td>8.01</td></tr><tr><td>CoTable+Semantic (GPT3.5)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/ meta-semantic classification</td><td>90.47</td><td>78.34</td><td>90.17</td><td>32.19</td><td>90.32</td><td>55.27</td></tr><tr><td>w/o meta-semantic classification</td><td>89.78</td><td>75.25</td><td>63.10</td><td>24.52</td><td>76.44</td><td>49.88</td></tr></table>

Ambiguous Questions Next, we tested the CoTable+Semantic IR model on ambiguous time-based questions. We did not test the other baseline models on this test since they performed poorly on the time-based questions in the unambiguous case, and thus will certainly fail in the ambiguous case. We test the CoTable+Semantic IR model first using only the original query (original query), which as expected, failed the task since it has no way to infer what is being referenced by the question. Next, we test the query rewrite method (query rewrite). This method performs similarly, though slightly worse, to the performance of the model that used the original unambiguous query. We also tested another baseline, which simply concatenated the previous sentences in the conversation chunk to the query and used the resulting conversation chunk as input to the IR system (context+query). This method performed surprisingly well achieving around $73 \%$ recall and 61 F2 with hMistral and around $7 7 \%$ recall and 65 F2 with GPT-3.5,. However, this is still significantly worse than the query rewrite.

Ablation Study One key addition we made to the CoTable method was the addition of semantic retrieval and the classifier, which classifies the query as either requiring meta-data or semantic based queries or both. This was motivated by preliminary tests on CoTable which found that a common mistake in the CoTable when function calling both meta-data and the content columns, was that the model got confused about when to do content based retrieval and when not to – models tended to perform content based search almost every query, even when content based retrieval was not appropriate. To provide empirical evidence the classifier helps mitigate this issue, we show results for a CoTable+Semantic IR model that does not use the meta-semantic classifier, but instead attempts to perform chain of table on the whole table, including the content column. The same prompts are used for this combined CoTable model, except now prompts show that there exists a content column that can be queried and have

![](images/63e757d23c79803c11156b5b98f82970ae3d6a1151a33ccf56ad2bad2e752e4e.jpg)  
Figure 4: Classification accuracy for the meta-semantic classifier on each test set.

related examples. Results are shown in table 3. We see that overall the meta-semantic classifier improves performance significantly for both Mistral and GPT-3.5. Interestingly, it helps these models in different cases: meta-semantic classification mainly helps Mistral with pure time questions, while it helps GPT-3.5 mainly in the Time+Content questions. This may be due to differences in the accuracy of the classifiers in each case (see figure 4). In the other cases, the classifier does not seem to effect performance significantly.

# 6 Limitations

We only tested our prompting on two models, Mistral-Hermes-7b and GPT-3.5-turbo. It is possible that other LLMs could behave differently with our prompts, though most of our prompts were derived from previous work that found these methods to be successful in other LLMS (e.g., [28]). We also did not test a variety of different semantic embedding models, and used a relatively small model (< 500 million parameters). It is possible that performance would be improved across the board with better embedding models, although it would be surprising if the relative performance of the baselines changed significantly given they all used the same retriever in our tests. Our dataset focused primarily on single-hop time-based questions, though out time+content dataset had multi-hop like questions. We also did not create ambiguous versions of the time+content based questions. Future work could focus on adding more complicated time-based questions and ambiguous versions of the time+content based questions.

# 7 Conclusion

Conversational AI agents may soon become common-place in our technologies, as they are integrated into virtual assistants, chat bots, and operating systems. An important capability for conversational agents is an understanding of when conversational events occurred in the past and an ability to handle ambiguous queries, which are commonplace in conversational contexts. We believe that our dataset and our system that combines chain-of-table and semantic retrieval methods makes a useful step in this direction.Our dataset provides a variety of questions that can test these ability, and our model provides strong foundation and baseline for RAG systems that can handle ambiguous and time based queries.Ultimately, long term memory systems for AI agents must combine the strengths of tabular

and database-centric information retrieval systems which have been developed and optimized for decades, with the semantic flexibility and additional affordances created by embeddings and LLMs.

# References

[1] Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36, 2024.   
[2] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pages 2206–2240. PMLR, 2022.   
[3] Aydar Bulatov, Yuri Kuratov, Yermek Kapushev, and Mikhail S Burtsev. Scaling transformer to 1m tokens and beyond with rmt. arXiv preprint arXiv:2304.11062, 2023.   
[4] David Castillo, Joseph Davidson, Finlay Gray, José Solorzano, and Marek Rosa. Introducing goodai ltm benchmark, 2024.   
[5] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.   
[6] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164, 2019.   
[7] Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, et al. Binding language models in symbolic languages. arXiv preprint arXiv:2210.02875, 2022.   
[8] Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. Trec cast 2019: The conversational assistance track overview. arXiv preprint arXiv:2003.13624, 2020.   
[9] Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. Cast 2020: The conversational assistance track overview. In In Proceedings of TREC, 2021.   
[10] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486, 2023.   
[11] Jihyoung Jang, Minseong Boo, and Hyounghun Kim. Conversation chronicles: Towards diverse temporal and relational dynamics in multi-session conversations. arXiv preprint arXiv:2310.13420, 2023.   
[12] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
[13] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.   
[14] Gibbeum Lee, Volker Hartmann, Jongho Park, Dimitris Papailiopoulos, and Kangwook Lee. Prompted llms as chatbot modules for long open-domain conversation. arXiv preprint arXiv:2305.04533, 2023.   
[15] Mosh Levy, Alon Jacoby, and Yoav Goldberg. Same task, more tokens: the impact of input length on the reasoning performance of large language models. arXiv preprint arXiv:2402.14848, 2024.

[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
[17] Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. Memochat: Tuning llms to use memos for consistent long-range open-domain conversation. arXiv preprint arXiv:2308.08239, 2023.   
[18] Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. arXiv preprint arXiv:2404.08801, 2024.   
[19] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024.   
[20] Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou, Haonan Chen, and Hongjin Qian. Large language models know your contextual search intent: A prompting framework for conversational search. arXiv preprint arXiv:2303.06573, 2023.   
[21] Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. Learning to relate to previous turns in conversational search. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1722–1732, 2023.   
[22] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22, 2023.   
[23] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305, 2015.   
[24] N. Reimers and O. Espejel, 2022.   
[25] Steve Tauroza and Desmond Allison. Speech rates in british english. Applied linguistics, 11(1):90–105, 1990.   
[26] teknium. Openhermes 2.5 - mistral 7b, 2023.   
[27] Yuqing Wang and Yun Zhao. Tram: Benchmarking temporal reasoning for large language models. arXiv preprint arXiv:2310.00835, 2023.   
[28] Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, et al. Chain-of-table: Evolving tables in the reasoning chain for table understanding. arXiv preprint arXiv:2401.04398, 2024.   
[29] Tianyu Wu, Shizhu He, Jingping Liu, Siqi Sun, Kang Liu, Qing-Long Han, and Yang Tang. A brief overview of chatgpt: The history, status quo and potential future development. IEEE/CAA Journal of Automatica Sinica, 10(5):1122–1136, 2023.   
[30] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.   
[31] Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. Large language models are versatile decomposers: Decompose evidence and questions for table-based reasoning. arXiv preprint arXiv:2301.13808, 2023.   
[32] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.

# A Appendix

# A.1 Code Availability

All code and data will be released upon publication. Our benchmark questions are included in the supplementary material.

# A.2 Experiment Details

In the appendix experiment description we say we run LLMs and sentence embedding locally on one L40 per run, unless using GPT-3.5. Our tests for the time-based queries, ambiguous queries, and time+content queries are run in essentially the same way. The models are presented with responses in chronological order one by one, similar to how they would be presented in a real conversational scenario. In addition to the response text, models are provided the date, timestamp, and speaker of the response. Models are not provided the session number, but rather must infer what session number it is. We switch to a new session if the time gap between the current and the previous response is greater than 20 minutes.

In all models the semantic embedder is used to store a single semantic vector per response. The Faiss library (CITE) is used to perform a flat search. The similarity measure used is cosine similarity. We use greedy sampling to generate text for chain-of-table and meta-semantic classification.

After all responses have been presented to the model, the current time is moved forward 50 minutes from the time of the last response. Then questions are presented to the model. The model is scored first on recall, which tells you the proportion of relevant response retrieved out of all relevant responses:

$$
\text {R e c a l l} = \frac {\text {R e l e v a n t r e t r i e v e d r e s p o n s e s}}{\text {A l l r e l e v a n t r e s p o n s e s}} \tag {1}
$$

Second, we score by the F2 score which is

$$
F 2 = \frac {5 \cdot \text {p r e c i s i o n} \cdot \text {r e c a l l}}{(4 * \text {p r e c i s i o n}) + \text {r e c a l l}} \tag {2}
$$

where

$$
\text {P r e c i s i o n} = \frac {\text {R e l e v a n t r e t r i e v e d r e s p o n s e s}}{\text {A l l r e t r i e v e d r e s p o n s e s}} \tag {3}
$$

The F2 is a harmonic mean of recall and precision, where recall is weighted more heavily than precision. The motivation for F2 rather than F1 is that an LLM cannot respond to a user with a fully correct answer if it is not given all the relevant information, so recall is essential. Prior work [15] has found empirically that LLMs can deal with some noise and irrelevant statements, but have been shown to perform worse on a variety of tasks when irrelevant text is present in its input, so precision should carry some weight but not as much as recall. Hence, we think F2 provides a reasonable measure of what it desired from an information retrieval system that provides input to an LLM.

Each individual test in the time-based query dataset consists of a different number of questions. To evenly weight each test when computing recall and F2 for these time based questions, we compute recall and F2 individually for each test, then average across tests.

# A.3 Time Test Question Descriptions

Here we describe each type of question that shows up in the time-based query tests.

• The earlier_today test poses variations on the question "what did we discuss earlier today?".   
• The date_span test poses variations of the question "what did we discuss between DATE1 and DATE2?"   
• The dates test poses variation of the question "what did we discuss on DATE?"   
• The day_span test poses variations of two questions "what did we talk about over the last three days?" and "what did we talked about over the last week?".

Table 4: Number of questions per test. We show the number questions for unambiguous (unambig.) and ambiguous (ambig.) questions, both when we sum together every question and when we sum together every version of every question (full).   

<table><tr><td>Test</td><td>Unambiguous. Qs</td><td>Unambiguous. Qs Full</td><td>Ambig. Qs</td><td>Ambig. Qs Full</td></tr><tr><td>earlier today</td><td>12</td><td>36</td><td>12</td><td>36</td></tr><tr><td>date span</td><td>180</td><td>2160</td><td>180</td><td>1080</td></tr><tr><td>dates</td><td>330</td><td>3960</td><td>330</td><td>1980</td></tr><tr><td>day span</td><td>24</td><td>108</td><td>24</td><td>108</td></tr><tr><td>last day</td><td>12</td><td>36</td><td>12</td><td>36</td></tr><tr><td>month</td><td>100</td><td>300</td><td>100</td><td>300</td></tr><tr><td>rel. day</td><td>317</td><td>938</td><td>304</td><td>912</td></tr><tr><td>rel. month</td><td>100</td><td>264</td><td>100</td><td>276</td></tr><tr><td>rel session</td><td>330</td><td>1014</td><td>330</td><td>1002</td></tr><tr><td>session span</td><td>258</td><td>1032</td><td>258</td><td>1032</td></tr><tr><td>session</td><td>294</td><td>1764</td><td>294</td><td>1764</td></tr><tr><td>time+content</td><td>177</td><td>-</td><td>-</td><td>-</td></tr><tr><td>total</td><td>2134</td><td>11612</td><td>1944</td><td>8526</td></tr></table>

• The last_day test poses variations of the question "what did we discuss last DAY NAME?", where DAY NAME can be Monday, Tuesday, etc.   
• The month test poses variations of the question " what did we discuss in MONTH, YEAR?"   
• The rel_day test poses variations of the question "what did we discuss N days ago?"   
• The rel_month test poses variations of the question "what did we discuss N months ago?"   
• The rel_session test poses variations of the question "what did we discuss N sessions ago?"   
• The session_span test poses variations of the question ’what did we talk about between session/conversation/discussion S1 and S2?’   
• The session test poses variations of the question "What did we discuss in session/discussion/conversation N?"

# A.4 Templates for Ambiguous Queries

The following are the templates used to create the surrounding context for the ambiguous time based questions. Here X is replaced with one of the temporal stamps noted in the previous section, e.g. "N session ago" or "on MONTH YEAR". These base templates a varied slightly as needed for each test, and further variations where made using different spellings of numerals.

# Template 1

speaker_a: I see in my calendar that we talked X.

speaker_b: Yes! we did talk then. I always enjoy our chats.

speaker_a: I enjoy them too! Can you summarize what we discussed?

# Template 2

speaker_a: I am try to remember what we talked about X. Can you remember that far back?

speaker_b: Yes, I can remember that far back.

speaker_a: Wow! You have a better memory than me!

speaker_b: I don’t know about that! But I can summarize our discussion for you if you’d like.

speaker_a: Yes, please do.

Table 5: Recall and F2 scores for ambiguous time-based questions.   

<table><tr><td rowspan="2">Retrieval Method</td><td rowspan="2">Query Type</td><td rowspan="2">Prompt Type</td><td colspan="2">hMistral7b</td></tr><tr><td>Recall</td><td>F2</td></tr><tr><td rowspan="2">CoTable+Semantic</td><td>Context+Qry</td><td>Original CoTb Prompt</td><td>73.51</td><td>61.59</td></tr><tr><td>Context+Qry</td><td>Context CoTb Prompt</td><td>74.56</td><td>52.19</td></tr></table>

# Template 3

speaker_a: I remember X we had several discussions.

speaker_b: Yes, we did.

speaker_a: But I cannot quite remember what we discussed.

speaker_b: Would you like me to tell you?

speaker_a: Yes, could you describe, in as much detail as you can, the content of those conversations?

# A.5 Prompting for Context Chunk to Chain-of-Table

In our ambiguous query test we found that of the methods we tried, disambiguating the query through a prompt based rewrite method worked the best. However, we found a simple baseline where we simply inject a chunk of previous context (i.e., the query plus the preceding 2-4 statements) significantly improves performance without changing our original chain of table prompts. To see if we could improve performance further, we augmented our chain of table prompt with examples that included chunks of the conversational context rather than just a single query/statement. Results are shown in table 5. We found this did not improve this method over the original prompt.

# A.6 Algorithm Description

Algorithm 1 Chain-of-Table w/ Semantic Retrieval   
Data: $(T,Q)$ ▷Table and query  
Result: $T_{new}$ ▷New Table, with only relevant rows  
function COTABLE+SEMANTIC(T,Q) $T_{new} = T$ chain = []  
do_meta, do_semantic = classify(Q)  
if do_meta then  
while $f \neq <\text{END}>$ do $\mathrm{f} = \text{get_function}(T_{\text{new}}, Q, \text{chain})$ $\arg1 = \text{get_arg1}(f, T_{\text{new}}, Q, \text{chain})$ $\arg2 = \text{get_arg2}(f, \arg1, T_{\text{new}}, Q, \text{chain})$ $T_{\text{new}} = \text{apply_func}(f, \arg1, \arg2, T_{\text{new}})$ chain.append([f, arg1, arg2])  
end while  
end if  
if do_semantic then: $T_{\text{new}} = \text{semantic_search}(T_{\text{new}}, Q)$ end if  
return $T_{\text{new}}$ end function

# A.7 Prompts

Below are descriptions of prompts used in our meta-semantic classifier, chain-of-table writer, and query rewriter.

# A.7.1 Meta-Classify Prompt

We have a table that stores a chat log of responses between two speakers in a table format. We store 1) semantic embeddings of the responses so we can search for responses with similar content, and 2) meta-data such as the times, dates, session number, and response number for each response. Each row stores information about one response. The columns in the table are:

/* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time */

We need to decide if the user’s query is referring to the meta-data of the chat log or not. You will be presented with a query from the user. Answer whether the query is referring to the meta-data of previous dialogues, such as the time, date, session or response number of previous portions of discussions. If the query is referring to meta-data, respond ’y’ for yes. If the query is not referring to meta-data, respond ’n’ for no. Only output a ’n’ or ’y’ character and nothing else. For example,

[EXAMPLES]

Query: [QUERY]

Output: [OUTPUT]

# A.7.2 Semantic-Classify Prompt

We have a table that stores a chat log of responses between two speakers in a table format. We store 1) semantic embeddings of the responses so we can search for responses with similar content, and 2) meta-data such as the times, dates, session number, and response number for each response. Each row stores information about one response. The columns in the table are:

```javascript
/\* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time \*/ 
```

We need to decide if the user’s query is referring to some specific topic or content, or if the query is only referring to meta-data or non-specific content. You will be presented with a query from the user. Answer whether the query is on some specific topic or content. If the query is referring specific topic or content, respond ’y’ for yes. If the query is not referring to a specific topic or content, respond ’n’ for no. Only output a $\nu _ { \mathrm { { n } } } ,$ , or ’y’ character and nothing else. For example,

[EXAMPLES]

Query: [QUERY]

Output: [OUTPUT]

# A.7.3 Function Write Prompt

The function write prompt writes the function name at each step of the chain of tables algorithm.

```txt
We have a table that stores a log of responses between two speakers in a table format. Information about speakers, response times, indexes, sessions, and dates are stored in the table, where each row stores information about one response. There are functions that allow us to isolate rows in the table. The columns in the table are: 
```

```javascript
/\* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time \*/ 
```

```txt
If the table only needs rows that have a certain value in a certain column to answer the question, we use 'f_value(column_name, [row_value1, row_value2, ...])' to select these rows. For example, 
```

```json
[EXAMPLES] 
```

```txt
If the table only needs rows with that have values within a certain range, to answer the question we use 'f_between(column_name, [min_value, max_value])' to select these rows. For example, 
```

```json
[EXAMPLES] 
```

```txt
If do not need to call a function, write <END>. 
```

```txt
Here are examples of using the operations to respond to the query. Be sure to end the operation chain after a few function calls with <END>. Do not repeat function calls. 
```

```json
[EXAMPLES] 
```

```txt
Current Date: [DATE] Current Time: [TIME] Current Session: [SESSION NUM]  
Query: [QUERY]  
Function Chain: [FUNCTION CHAIN SO FAR] -> [OUTPUT] 
```

# A.7.4 Argument 1 Write Prompt for f_value

The first argument the f_value function is written by an LLM using the following prompt.

We have a table that stores a log of responses between two speakers in a table format. Information about response speaker names, and dates are stored in the table, where each row stores information about one response. There are functions that allow us to isolate rows in the table. The columns in the table are:

/* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time */

If the table only needs rows that have a certain value in a certain column to answer the question, we use ’f_value(column_name, [row_value1, row_value2, ...])’ to select these rows. For example,

[EXAMPLES]

Here are examples of chaining together multiple function calls. Each chain of function calls is ended when the model outputs <END>. For example,

[EXAMPLES]

Finish the following function chain. Do not write any extra text. Only output the first argument of the ’f_value’ function.

Current Date:[DATE] Current Time:[TIME] Current Session:[SESSION NUM] Query: [QUERY]

Function Chain: [FUNCTION CHAIN SO FAR] -> [OUTPUT]

# A.7.5 Argument 1 Write Prompt for f_between

The first argument the f_value function is written by an LLM using the following prompt.

We have a table that stores a log of responses between two speakers in a table format. Information about response speaker names, and dates are stored in the table, where each row stores information about one response. There are functions that allow us to isolate rows in the table. The columns in the table are:

/* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time */

If the table needs rows between two values in a certain column, we use ’f_between(column_name, [min_value, max_value])’ to select these rows. For example,

[EXAMPLES]

Here are examples of chaining together multiple function calls. Each chain of function calls is ended when the model outputs <END>. For example,

[EXAMPLES]

Finish the following function chain. Do not write any extra text. Only output the first argument of the ’f_between’ function.

Current Date:[DATE] Current Time:[TIME] Current Session:[SESSION NUM] Query: [QUERY] Function Chain: [FUNCTION CHAIN SO FAR] -> [OUTPUT]

# A.7.6 Argument 2 Write Prompt for f_value

The first argument the f_value function is written by an LLM using the following prompt.

We have a table that stores a log of responses between two speakers in a table format. Information about response speaker names, and dates are stored in the table, where each row stores information about one response. There are functions that allow us to isolate rows in the table. The columns in the table are:

```javascript
/\* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time \*/ 
```

We are in the process of isolating the rows relevant to the query through a series of function calls. Our next step is to decide which rows to isolate using the f_value function. The f_value function returns all rows that have a value equal to at least one of the given values, i.e., f_value(column_name, [value_1, value_2,...]). Here are examples of how to fill in the values for the second argument:

```json
[EXAMPLES] 
```

Fill in the remaining argument of the f_value function. Do not explain your answer, just provide the missing values. Current Date:[DATE] Current Time:[TIME] Current Session:[SESSION NUM] Query: [QUERY] Function Chain: [FUNCTION CHAIN SO FAR] Answer: [OUTPUT]

# A.7.7 Argument 2 Write Prompt for f_between

The first argument the f_value function is written by an LLM using the following prompt.

We have a table that stores a log of responses between two speakers in a table format. Information about response speaker names, and dates are stored in the table, where each row stores information about one response. There are functions that allow us to isolate rows in the table. The columns in the table are:

```javascript
/\* Response_Index | Session_Index | Speaker | Day_Name | Week | Date | Time \*/ 
```

We are in the process of isolating the rows relevant to the query through a series of function calls. Our next step is to decide which rows to isolate using the f_between function. The f_between function returns all rows that have a value between a given minimum and maximum value for a given column, i.e., f_between(column_name, [min_value, max_value]). Here are examples of how to fill in the values for the second argument:

```json
[EXAMPLES] 
```

Fill in the remaining argument of the f_between function. Do not explain your answer, just provide the missing values. Current Date:[DATE] Current Time:[TIME] Current Session:[SESSION NUM] Query: [QUERY] Function Chain: [FUNCTION CHAIN SO FAR] Answer: [OUTPUT]

# A.7.8 Query Rewrite Prompt

Please help reformulate the question into a rewrite that can fully express the user’s information needs without the need of context and while removing irrelevant sentences. Here are several example dialogues, where each turn contains a portion of dialogue and a rewrite.

[EXAMPLES]

If the query is not ambiguous or if there are no preceding sentences, just repeat the question. For example,

[EXAMPLES]

Query: [QUERY]

Rewrite: [OUTPUT]