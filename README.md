#GDS Hackpions 2.0
EY GDS Hackpions 2.0, the third edition of the hackathon from the Enablement Services function of EY Global Delivery Services (GDS) — the hottest virtual event of 2021!

The hackathon aims to connect participants with experts, to tackle real-world challenges. It brings together technology champions, students and working professionals such as yourself, from varied domains. Through the course of the event, we’d encourage you to gather unique ideas and innovation, that can help alleviate business problems faced by EY GDS teams during and after the pandemic. Join this unique and fun upskilling experience, to showcase your coding craftsmanship.

In this virtual hackathon event, we’ll be asking you to develop tools, apps, bots and other advanced solutions. You can partner with hackers from across the world and build hacks in critical and extensive domains, by leveraging technologies such as artificial intelligence, cloud and data analytics. You’ll also get a chance to be mentored by professionals from EY GDS and HackerEarth, who will handhold you while you take up the challenge to turn hacks into real-world business solutions.

## NLP-based tagging solution
The Intelligent Automation team manually processes a large volume of exception data. The manual process includes examining the exception reasons and categorizing them broadly into business or technical issues, which is time consuming and increases turnaround time. 

You need to build an intelligent solution, to auto label exceptions as they occur.

Deliverables

Propose and design a machine learning algorithm, that will carry out the following tasks:

Use training data and keywords across categories, for each of the exceptions, as given below:
Create an input excel file, with a list of exceptions.
* Business exception
* System exception
* Other

## Usage
run `pip install -r requirements.txt` to install libraries

train model `python train.py`
predict model `python predict.py -s <STR: sentence to predict>`