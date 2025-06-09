
# Preprocess MRI Image
def preprocess_mri_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Set page config
st.set_page_config(page_title="Alzheimer's Stage Predictor", page_icon="🧠", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 Alzheimer's Stage Prediction</h1>
    <p style='text-align: center;'>Upload an MRI scan and enter clinical data to detect the stage of Alzheimer's disease.</p>
    <hr style='border: 1px solid #ccc;'/>

""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app uses a multimodal deep learning model to classify Alzheimer's stages based on:

- MRI brain image (grayscale)
- Clinical features like MMSE, eTIV, nWBV, etc.
""")

# Upload MRI
st.subheader("📤 Upload MRI Image")
mri_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

# Clinical input fields
st.subheader("🩺 Enter Clinical Data")
col1, col2 = st.columns(2)

with col1:
    mmse = st.number_input("MMSE Score (0–30)", min_value=0.0, max_value=30.0)
    etiv = st.number_input("eTIV")
    nwbv = st.number_input("nWBV")
    asf = st.number_input("ASF")

with col2:
    age = st.number_input("Age", min_value=0)
    ses = st.number_input("SES (0–5)", min_value=0.0, max_value=5.0)
    gender = st.radio("Gender", ["Male", "Female"])
    gender_encoded = 1 if gender == "Male" else 0

st.markdown("<hr/>", unsafe_allow_html=True)

# Prediction button
if st.button("🧠 Predict Alzheimer's Stage"):
    if mri_file is None:
        st.error("🚨 Please upload an MRI image before proceeding.")
    else:
        try:
            # Preprocess data
            clinical_input = np.array([[mmse, etiv, nwbv, asf, age, ses, gender_encoded]])
            clinical_scaled = scaler.transform(clinical_input)
            mri_input = preprocess_mri_image(mri_file)

            # Model prediction
            prediction = model.predict([mri_input, clinical_scaled])
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_cdr = label_encoder.inverse_transform([predicted_class])[0]

            stage_name, stage_desc = cdr_stage_mapping.get(predicted_cdr, ("Unknown", "Unknown"))

            # Display result
            st.success(f"🎯 *Predicted Stage: {stage_name} - *{stage_desc} (CDR: {predicted_cdr})")

            if predicted_cdr == 0.0:
                st.balloons()
                st.info("🧘 You are currently in a healthy state. Keep it up!")

        except Exception as e:
            st.error(f"❌ Prediction failed due to an error:\n\n{str(e)}")

# Footer
st.markdown("""
    <hr/>
    <div style='text-align: center;'>
        <small>Developed with ❤ using Streamlit | Multimodal Deep Learning</small>
    </div>
""", unsafe_allow_html=True)
