from joblib import load

# Load the model and vectorizer
model = load('voting_classifier_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

def classify_text(input_text):
    # convert to tfidf
    input_vectorized = vectorizer.transform([input_text])
    
    prediction = model.predict(input_vectorized)
    
    return prediction[0]

###Example usage###

input_string = """the scene is in the  in the kitchen
the mother is wiping dishes and the water is running on the
a child is trying to get  a boy is trying to get cookiesoutta  out of a jar and hes about to tip over on a stool
uh the little girl is reacting to his falling
uh it seems to be summer out
the window is open
the curtains are blowing
it must be a gentle breeze
theres grass outside in the garden
uh mothers finished certain of the  the dishes
kitchens very tidy
the mother seems to have nothing in the house to eat except cookiesin the cookie jar
uh the children look to be almost about the same size
perhaps theyre twins
theyre dressed for summer warm weather
um you want more
 the mothers in a short sleeve dress
 ill hafta say its warm"""

classification = classify_text(input_string)
print("The classification is:", classification)
