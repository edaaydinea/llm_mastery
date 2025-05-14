# LLM Mastery: ChatGPT, Gemini, Claude, Llama3, OpenAI & APIs

Learn to master the latest in Large Language Models (LLMs) including ChatGPT, Gemini, Claude, Llama3, and OpenAI APIs. This course provides comprehensive insights and practical applications.

[Course Link](https://www.udemy.com/course/llm-mastery-chatgpt-gemini-claude-llama3-openai-apis)

## Table of Contents

- [LLM Mastery: ChatGPT, Gemini, Claude, Llama3, OpenAI \& APIs](#llm-mastery-chatgpt-gemini-claude-llama3-openai--apis)
  - [Table of Contents](#table-of-contents)
  - [Day 1: How LLMs Work: Parameters, Weights, Inference, Neural Networks and More](#day-1-how-llms-work-parameters-weights-inference-neural-networks-and-more)
  - [Day 2: Additional Capabilities of LLMs \& Future Developments](#day-2-additional-capabilities-of-llms--future-developments)
  - [Day 3: Prompt Engineering: Effective Use of LLMs in the Standard Interface](#day-3-prompt-engineering-effective-use-of-llms-in-the-standard-interface)
  - [Day 4: LLM Customization: System Prompts, Memory, RAG \& Creating Expert Models or GPTs](#day-4-llm-customization-system-prompts-memory-rag--creating-expert-models-or-gpts)
  - [Day 5: Closed-Sourse LLMs: An Overview of Available Models and how to use them](#day-5-closed-sourse-llms-an-overview-of-available-models-and-how-to-use-them)
  - [Day 6: APIs of Closed-Source LLMs](#day-6-apis-of-closed-source-llms)
  - [Day 7: Open-Source LLMs: Available Models and Their Use in the Claude \& Locally](#day-7-open-source-llms-available-models-and-their-use-in-the-claude--locally)
  - [Day 8: First Steps to Creating Your Own Apps via APIs in Google Colab](#day-8-first-steps-to-creating-your-own-apps-via-apis-in-google-colab)
  - [Day 9: Apps, Chatbots, and AI Agents: Build cool Stuff for Automation and FUN](#day-9-apps-chatbots-and-ai-agents-build-cool-stuff-for-automation-and-fun)
  - [Day 10: LLM Security: Copyrights, Rights, Jailbreaks, and Prompt Injections](#day-10-llm-security-copyrights-rights-jailbreaks-and-prompt-injections)
  - [Day 11: Comparison Test: What is the Best LLM?](#day-11-comparison-test-what-is-the-best-llm)
  - [Day 12: Finetuning LLMs: OpenAI API and Google Colab](#day-12-finetuning-llms-openai-api-and-google-colab)

## Day 1: How LLMs Work: Parameters, Weights, Inference, Neural Networks and More

**What I did today:**

- Reviewed the basics of Language Models (LMs) including pre-training and fine-tuning.
- Learned about the transformer architecture and reinforcement learning.
- Explored the components of an LM: parameter file and run file.
- Understood the process of creating parameter files using GPUs and text data.
- Studied neural networks and their functioning with forward and back propagation.
- Examined how neural networks work with word tokens in LLMs.
- Discussed the transformer architecture and its current limitations.
- Learned about the mixture of experts approach for transformer architecture.
- Reviewed the fine-tuning process to create assistant models.
- Understood reinforcement learning and its application in LLMs.
- Discussed the scaling laws of LLMs and the importance of GPU and data.

**Resources**:

- [day1 notes.ipynb](./notes/day1.ipynb)

## Day 2: Additional Capabilities of LLMs & Future Developments

**What I did today:**

- Explored the multimodal capabilities of LLMs, including processing images, audio, and video.
- Learned about ChatGPT's ability to use various tools like calculators and Python libraries.
- Studied the vision capabilities of LLMs, including image recognition and visual processing.
- Discussed the potential for LLMs to engage in natural, conversational speech.
- Reviewed the concept of System 1 and System 2 thinking in LLMs.
- Learned about recent updates to ChatGPT, including real-time web searches and content creation tools.
- Examined the advancements in OpenAI's O3 model towards AGI.
- Discussed the concept of self-improvement in AI inspired by AlphaGo.
- Explored methods for improving LLM performance, such as RAG and prompt engineering.
- Envisioned the future of LLMs as comprehensive operating systems.

**Resources**:

- [day2 notes.ipynb](./notes/day2.ipynb)

## Day 3: Prompt Engineering: Effective Use of LLMs in the Standard Interface

**What I did today:**

- Gained an understanding of the fundamental principles of prompt engineering, applicable across various Large Language Models (LLMs).
- Identified common LLM interfaces such as ChatGPT, Hugging Chat, Copilot, and Gemini, noting their shared basic functionalities.
- Learned about token limits in LLMs and their importance in managing context within conversations.
- Recognized how different prompt phrasings can significantly impact an LLM's response, underscoring the importance of effective prompt engineering.
- Grasped the concept of semantic association and how words trigger related concepts in LLMs, influencing their responses.
- Explored the use of structured prompts, consisting of modifiers and topics, to guide LLMs towards more accurate and tailored outputs.
- Discovered the effectiveness of instruction prompting and the use of specific phrases like "Let's think step by step" to enhance LLM performance.
- Understood how role prompting can leverage semantic association by assigning specific roles to LLMs, resulting in more relevant responses.
- Learned about zero-shot, one-shot, and few-shot prompting techniques, which utilize examples to guide LLM outputs.
- Investigated reverse prompt engineering as a method to extract underlying prompts from text to replicate desired styles and content.
- Explored chain of thought prompting, which involves providing step-by-step reasoning to improve the accuracy of LLM responses, especially for complex tasks.
- Became familiar with the Tree of Thought prompting technique, a more advanced method for complex problem-solving involving the generation and evaluation of multiple solution paths.
- Reviewed the diverse applications of LLMs across various domains, emphasizing the importance of effective prompting to maximize their potential.
- Learned how to combine different prompting techniques, such as role prompting, structured prompts, and few-shot prompting, to achieve optimal LLM outputs, using a real-world example.
- Recapped the various prompt engineering techniques covered, including token limits, semantic association, and different prompting strategies, highlighting the importance of practical application.

**Resources**:

- [day3 notes.ipynb](./notes/day3.ipynb)

## Day 4: LLM Customization: System Prompts, Memory, RAG & Creating Expert Models or GPTs

**what I did today:**

- Explored fundamental methods for chatbot personalization, with a focus on the user-friendly interface of ChatGPT.
- Investigated the customization of ChatGPT's memory feature to retain user preferences and context across conversations.
- Examined the use of custom instructions as persistent system prompts for tailoring ChatGPT's responses based on personal context and desired behavior.
- Understood the concept of in-context learning as a short-term memory solution for language models.
- Learned about Sparse Prime Representation (SPR) as a more token-efficient approach to in-context learning.
- Gained knowledge of Retrieval-Augmented Generation (RAG) technology for enabling long-term memory in language models through vector databases.
- Discovered how to implement simplified RAG using custom GPTs within ChatGPT by uploading external knowledge files.
- Navigated the ChatGPT GPT Store, exploring various custom GPTs for code generation, PDF analysis, and YouTube video summarization.
- Identified three potential strategies for monetizing custom GPTs: revenue sharing, lead generation, and facilitating upsells.
- Understood the necessity of creating a builder profile in ChatGPT to potentially monetize GPTs and generate leads.
- Learned the process of creating a custom GPT with specific knowledge to generate leads and facilitate upsells by strategically incorporating links.
- Defined Application Programming Interfaces (APIs) as software components enabling communication between different applications.
- Explored the integration of Zapier actions into GPTs to automate tasks across various platforms like Gmail and Google Docs.
- Demonstrated the process of integrating external APIs into custom GPTs using the Cat API as a practical example.
- Synthesized the key learnings from the section, emphasizing the application of customization techniques for personalizing ChatGPT and leveraging GPTs for business purposes, particularly lead generation.

**Resources**:

- [day4 notes.ipynb](./notes/day4.ipynb)

## Day 5: Closed-Sourse LLMs: An Overview of Available Models and how to use them

**What I did today:**

- Analyzed video transcripts summarizing key aspects of various closed-source Large Language Models (LLMs) and related platforms.
- Identified the core functionalities, advantages, and disadvantages of models such as OpenAI's ChatGPT, Google's Gemini, and Anthropic's Claude.
- Summarized the features and use cases of platforms like Perplexity and Poe, which build upon these underlying LLMs.
- Understood the integration of OpenAI's technology into Microsoft's Copilot across various Microsoft 365 applications.
- Noted the specific capabilities of Copilot within Word, PowerPoint, Outlook, and Excel, as well as the introduction of Copilot GPTs.
- Reviewed GitHub Copilot as an AI-powered tool for programmers and its potential impact on coding efficiency.
- Synthesized the overall value proposition of Microsoft Copilot and its associated subscription models, comparing it to free alternatives.
- Compiled a summary of the key learnings regarding the discussed closed-source LLMs and their respective platforms.

**Resources**:

- [day5 notes.ipynb](./notes/day5.ipynb)

## Day 6: APIs of Closed-Source LLMs

**What I did today:**

- Gained foundational knowledge of working with closed-source LLM APIs, with a specific focus on the OpenAI API's core concepts, playground usage, and billing structure.
- Acquired hands-on experience navigating and experimenting within the OpenAI API playground, testing various models and interface modes (Chat, Assistants, Completions).
- Mastered the crucial steps for setting up API billing accounts and configuring usage limits to enable access to the latest models and manage costs effectively.
- Explored advanced model control settings in the OpenAI playground, including Temperature, Maximum Tokens, Stop Sequences, Top P, Frequency Penalty, and Presence Penalty.
- Learned to inspect the underlying code generated by playground interactions, facilitating the transition from experimentation to building custom applications.
Developed an understanding of the pay-per-token pricing models for various closed-source APIs and compared the costs and features of OpenAI, Google Gemini, and Anthropic.
- Investigated the capabilities of the Google Gemini API, including its multimodal features and Vertex AI integration.
Evaluated the Anthropic API and its Claude models, assessing their competitiveness against other offerings based on pricing and features.
Concluded that the OpenAI API currently offers a strong combination of model quality, competitive pricing, and robust features, positioning it as the preferred choice for immediate development tasks.
- Prepared to transition the focus to exploring open-source language models for broader application possibilities.

**Resources:**

- [day6 notes.ipynb](./notes/day6.ipynb)

## Day 7: Open-Source LLMs: Available Models and Their Use in the Claude & Locally

**What I did today:**

- Gained a comprehensive understanding of open-source Large Language Models (LLMs), including their benefits such as cost-effectiveness, local/cloud deployment flexibility, and customization options like fine-tuning and creating uncensored versions.
- Explored the Hugging Face platform as a central hub for accessing a vast repository of open-source models, datasets, and essential libraries (Transformers, Datasets, Tokenizers), and understood its role in democratizing AI.
- Investigated Hugging Chat as a free, cloud-based interface for interacting with diverse open-source LLMs, noting its ChatGPT-like UI, function calling capabilities (web search, image generation), and the ability to create custom AI assistants.
- Learned about Groq and its innovative Language Processing Unit (LPU) technology, which enables exceptionally fast inference speeds for LLMs, offering low-latency interactions with various open-source models.
- Understood the process and hardware requirements (GPU, CPU, RAM, CUDA) for running LLMs locally, highlighting LM Studio as a user-friendly application for downloading and managing local models.
- Practically explored using open-source models like Llama 3, Mistral, and Phi-3 within LM Studio, including model discovery, download, and configuration of inference parameters.
- Analyzed the concepts of bias and censorship in LLMs, and learned how uncensored fine-tuned models like "Dolphin" versions of Llama 3 offer unrestricted responses, emphasizing the importance of responsible use.
- Successfully learned to set up a local HTTP server using LM Studio, exposing downloaded LLMs through an OpenAI-compatible API for custom application development, with example code provided.
- Gained insights into the process of fine-tuning open-source LLMs using Hugging Face AutoTrain and Google Colab, considering the associated costs, time commitment, and the alternative of leveraging pre-fine-tuned models.
- Reviewed the capabilities of Grok by xAI, noting its large context window and multimodal strengths, while also understanding the practical limitations of running its open-source version (Grok-1) locally due to its size.
- Received an update on Llama 3.1, highlighting its strong performance across its 8B, 70B, and 405B parameter models, its accessibility via cloud APIs (including Groq's fast API), and local execution options with Ollama.
- Learned about the DeepSeek-R1 model, its "Test-Time Compute" feature for enhanced reasoning, its open-source MIT license, and the availability of distilled versions rivaling OpenAI-o1-mini.
- Consolidated understanding of the open-source LLM landscape, emphasizing the value of local execution for data privacy and uncensored access, and previewed future topics including Ollama and advanced application development.

**Resources:**

- [day7 notes.ipynb](./notes/day7.ipynb)

## Day 8: First Steps to Creating Your Own Apps via APIs in Google Colab

**What I did today:**

- Successfully set up and utilized Google Colab as an interactive environment for executing Python code and interfacing with OpenAI APIs, including library installation and runtime configuration.
- Gained foundational knowledge of GitHub as a code repository and community resource, including basic navigation and the process for account creation.
- Mastered the core workflow for interacting with various OpenAI APIs, including obtaining and securely managing API keys, and referencing official documentation for implementation.
- Implemented API calls for text generation with OpenAI models (e.g., GPT-4o), covering prompt engineering, model selection, and techniques for saving outputs, including leveraging ChatGPT for code refinement.
- Integrated DALL-E 3 via the OpenAI API within Google Colab to programmatically generate images from text prompts, including enhancing code to display and download the generated images.
- Utilized the OpenAI Text-to-Speech (TTS) API in Colab to convert text into natural-sounding audio, managing model and voice selection, and saving audio output as .mp3 files.
- Employed the OpenAI Whisper API for speech-to-text transcription within Colab, including managing audio file uploads and correctly referencing file paths for API processing.
- Explored OpenAI's Vision API capabilities using multimodal models (e.g., GPT-4o) in Colab to analyze and describe images provided via URLs.
- Developed a comprehensive Google Colab notebook serving as a personal toolkit for interacting with multiple OpenAI API functionalities (text, image, TTS, speech-to-text, vision), reinforcing best practices like using API key placeholders and saving personal copies.
- Acquired practical experience in troubleshooting API calls and enhancing code functionality with assistance from AI tools like ChatGPT.

**Resources:**

- [day8 notes.ipynb](./notes/day8.ipynb)
- [day8 codes.ipynb](./codes/day8.ipynb)

## Day 9: Apps, Chatbots, and AI Agents: Build cool Stuff for Automation and FUN

**What I did today:**

- Gained a foundational understanding of AI agents, differentiating them from basic chatbots by their autonomous task performance, decision-making capabilities, and ability to interact with environments, often through a supervisor LLM coordinating sub-expert LLMs.
- Explored the core components of AI agents, including Natural Language Processing (NLP), Large Language Models (LLMs), and vector databases, and identified their wide range of applications from customer service to autonomous systems.
- Reviewed key development platforms and frameworks for AI agents, such as LangChain, FlowiseAI, LangFlow, and Vector Shift, with a focus on their usability and cost-effectiveness.
- Acquired hands-on experience with Vector Shift, a no-code, cloud-based platform built on LangChain, by constructing a basic chatbot pipeline using input, OpenAI LLM, and output nodes.
- Enhanced a simple Vector Shift chatbot to a Retrieval Augmented Generation (RAG) system by integrating a custom knowledge base, enabling it to answer questions based on uploaded data like PDFs or website content.
- Investigated various methods for deploying Vector Shift RAG chatbots, including as standalone applications, embedded search bars or chat bubbles on websites (HTML, WordPress), and integrations with WhatsApp and Slack.
- Designed a multi-LLM AI agent architecture in Vector Shift, where a classifier LLM routes user queries to specialized expert LLMs, each trained on distinct knowledge bases.
- Transitioned from paid platforms to open-source tools, identifying LangChain as a core framework and FlowiseAI as a user-friendly alternative for creating AI agents with a drag-and-drop interface.
- Successfully set up a local Flowise development environment using Node.js, including installation, starting the server, and understanding update processes.
- Became proficient in navigating the Flowise UI, including creating chat/agent flows, utilizing the marketplace for templates, managing credentials, and integrating tools like SerpApi and assistants for function calling.
- Developed a RAG chatbot within Flowise using the "Conversational Retrieval Q&A chain" template, integrating components like GPT-3.5 Turbo, Pinecone for vector storage, and OpenAI embeddings.
- Leveraged the OpenAI Assistant API within Flowise to build a powerful AI assistant from scratch, equipping it with tools such as a calculator, web search (SerpApi), a code interpreter, and RAG capabilities via file search.
- Installed Ollama for local LLM hosting, downloaded Llama 3, and successfully ran it on a local server, enabling private and cost-effective model usage.
- Constructed a local RAG chatbot in Flowise using Llama 3 via Ollama for both chat model and embeddings, combined with an in-memory vector store and a Cheerio web scraper for document loading.
- Developed multi-agent AI systems in Flowise, creating a "Supervisor-Worker" architecture for tasks like software development (Snake game example) and content creation (social media agent, math agent), assigning specific tools and different LLMs (OpenAI GPT-4o, local Ollama/Llama 3) to various workers.
- Learned to run Flowise agent flows as standalone applications on a local computer using the "Share Chatbot" feature.
- Mastered deploying Flowise to the cloud using Render, including forking the GitHub repository, configuring web services, and setting up environment variables and persistent disks for both free and paid plans.
- Acquired skills in embedding Flowise chatbots into websites (HTML, WordPress, Shopify) using JavaScript snippets, with advanced customization of the chat bubble's appearance and behavior.
- Explored advanced Flowise features, including viewing message history, enabling user feedback, setting starter prompts, implementing rate limits, integrating speech-to-text (OpenAI Whisper), capturing leads, and utilizing Flowise API endpoints with API keys.
- Experimented with building a free chatbot in Flowise using open-source models (Mixtral 8x7B Instruct) via Hugging Face inference endpoints.
- Significantly improved inference speed for local RAG chatbots and AI agents by integrating Groq's API for LLM inference (Llama 3), while retaining existing embedding solutions.
- Recognized the value of the Flowise Marketplace for discovering pre-built chains and templates, such as If/Else logic and various agent systems, to accelerate development.
- Briefly reviewed alternative AI agent frameworks like Microsoft's Autogen, CrewAI, and Agency Swarm, concluding that Flowise combined with Langchain offers a more practical solution for most users.
- Consolidated overall learning on AI agents and chatbot development, covering definitions, platform exploration (VectorShift, Flowise), diverse chatbot creation (basic, RAG, multi-agent), local and cloud deployment, and advanced feature utilization.

**Resources:**

- [day9 notes.ipynb](./notes/day9.ipynb)

## Day 10: LLM Security: Copyrights, Rights, Jailbreaks, and Prompt Injections

**What I did today:**

- Gained a comprehensive understanding of critical Large Language Model (LLM) vulnerabilities, including diverse jailbreaking techniques (many-shot, zero-shot, prefix injection, obfuscated inputs, adversarial suffixes, visual methods), indirect prompt injection via external data sources, and data poisoning attacks during various training phases.
- Explored the mechanisms behind LLM attacks, such as how hidden instructions in external content can override an LLM's original purpose, and how trigger-based behaviors can be embedded through corrupted training data.
- Assessed the current landscape of copyright for AI-generated content, including OpenAI's "Copyright Shield" for specific users, Midjourney's ownership policies, Adobe Firefly's "commercially safe" design approach, and the user responsibilities associated with open-source models like Stable Diffusion.
- Delved into data privacy considerations across various AI platforms, differentiating data usage policies for standard cloud-based LLMs (e.g., ChatGPT free/Plus, Bard, Copilot) versus more private options like enterprise plans, API access (OpenAI API), or locally hosted open-source models.
- Investigated platform-specific rules regarding AI-generated content and critically evaluated the demonstrated unreliability of current AI detection tools, understanding the challenges this poses for content creators.
- Studied core principles for ethical AI development and deployment, including transparency, fairness, non-maleficence, accountability, and privacy, and considered their application in areas like autonomous vehicles and business decision-making.
- Recognized significant societal risks associated with AI, such as the amplification of existing biases from training data, potential for misuse in creating deepfakes or cyber-attacks, severe consequences of incorrect AI decisions in critical applications, and the concern of human skill degradation due to over-reliance.
- Learned practical strategies for mitigating AI risks, such as scrutinizing LLM requests for personal information, verifying AI-generated content before commercial use, employing enhanced privacy options (APIs, local hosting) for sensitive data, and applying substantial human editing to AI-assisted content for restrictive platforms.
- Acknowledged the dynamic "cat and mouse game" of AI security, underscoring the need for continuous vigilance, critical thinking, and adapting behavior based on new learnings in the evolving AI landscape.
- Understood the expanding capabilities of LLMs, including function calling to integrate with external tools like diffusion models (e.g., DALL-E, Stable Diffusion) for multimedia generation, and the associated broadening of the attack surface and ethical considerations.

**Resources:**

- [day10 notes.ipynb](./notes/day10.ipynb)

## Day 11: Comparison Test: What is the Best LLM?

**What I did today:**

**Resources:**

## Day 12: Finetuning LLMs: OpenAI API and Google Colab

**What I did today:**

**Resources:**
