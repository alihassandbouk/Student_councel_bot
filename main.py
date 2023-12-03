import streamlit as st
import langchain_helper as lgh
from PIL import Image

#putting image
img = Image.open("student_councel.png")
st.image(image=img,width=500, )

#title and description
st.title("Student Councel Assistant Bot  🤖🎓")
st.markdown('''
    ## This App answers the common questions asked by ULFG3 students
            
    * #### **Done Using:** Streamlit, Palm2, Langchain
            
    ***
''')

#input field
st.header("Enter your question:")
question = st.text_input("")

#print results
answer = ""
if question:
    chain = lgh.generate_chain()
    answer = chain(question)
    st.subheader("Answer:")
    st.write(answer["result"])
