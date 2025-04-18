import streamlit as st
import os

class Sidebar():
    def __init__(self) -> None:
        self.model_name = None
        self.confidence_threshold = None
        self.detection_type = None
        self.title_img = self.get_asset_path('ai_med.jpg')

        self._titleimage()
        self._detection_type()
        self._model()
        self._confidencethreshold()

    def get_asset_path(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'images', filename)

    def _titleimage(self):
        st.sidebar.image(self.title_img)

    def _detection_type(self):
        st.sidebar.markdown('## Step 1 : Choose Detection Type')
        self.detection_type = st.sidebar.selectbox(
            label='Which type of detection ?',
            options=[
                'Bone Fracture Detection',
                'Alzheimer Detection'
            ],
            index=0,
            key='detection_type'
        )

    def _model(self):
        st.sidebar.markdown('## Step 2 : Choose Model')

        model_options = []

        if self.detection_type == 'Bone Fracture Detection':
            model_options = [
                'YoloV8',
                'FastRCNN with ResNet',
                'VGG16'
            ]
        elif self.detection_type == 'Alzheimer Detection':
            model_options = [
                'CNN Alzheimer v1',
                'ResNet Alzheimer v2',
                'Custom Alzheimer Model'
            ]

        self.model_name = st.sidebar.selectbox(
            label='Available Models :',
            options=model_options,
            index=0,
            key='model_name'
        )

    def _confidencethreshold(self):
        st.sidebar.markdown('## Step 3 : Set Confidence Threshold')
        self.confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.00, 1.00, 0.5, 0.01)
