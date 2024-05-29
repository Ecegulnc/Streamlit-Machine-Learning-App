import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce
import seaborn as sns

st.title("Streamlit Machine Learning App")

st.write("""
# Explore different classifiers and datasets
Which one is the best?
""")

dataset_source = st.sidebar.selectbox("Select Data Source", ("Use built-in dataset", "Upload your own CSV"))

if dataset_source == "Use built-in dataset":
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine", "Digits"))
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write("Shape of dataset:", user_data.shape)
        if st.sidebar.checkbox("Preview data"):
            st.write(user_data.head())
        dataset_name = "User uploaded"
    else:
        st.warning("Please upload a CSV file.")
        dataset_name = None

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boosting", "AdaBoost", "Naive Bayes"))
scaling_method = st.sidebar.selectbox("Select Feature Scaling", ("None", "StandardScaler", "MinMaxScaler", "RobustScaler"))

# Add option for data visualization
visualize_data = st.sidebar.checkbox("Visualize data")

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    else:
        data = None
    if data:
        X = pd.DataFrame(data.data, columns=data.feature_names)  # Convert to DataFrame
        y = data.target
    else:
        # Updated the column selection to exclude the target column
        X = user_data.iloc[:, 1:]  # Exclude target column (assumed to be the first column)
        y = user_data.iloc[:, 0]   # Target column as y
        
        # Handle non-numeric data
        if X.select_dtypes(include=['object']).shape[1] > 0:
            encoder = ce.OneHotEncoder(use_cat_names=True)
            X = encoder.fit_transform(X)
        # Encode target if it's not numeric
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
    return X, y

if dataset_name:
    X, y = get_dataset(dataset_name)
    if dataset_name != "User uploaded":
        st.write("Shape of dataset:", X.shape)
        st.write("Number of classes:", len(np.unique(y)))

    # Visualize the dataset
    if visualize_data:
        plot_type = st.sidebar.selectbox("Select plot type", ("Histogram", "Box Plot", "Pair Plot", "Correlation Heatmap"))

        if plot_type == "Histogram":
            st.write("Histograms")
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                fig, ax = plt.subplots()
                ax.hist(X[col], bins=30, color='blue', alpha=0.7)
                plt.title(col)
                st.pyplot(fig)

        elif plot_type == "Box Plot":
            st.write("Box Plots")
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=y, y=X[col], ax=ax)
                plt.title(col)
                st.pyplot(fig)

        elif plot_type == "Pair Plot":
            st.write("Pair Plot")
            df = pd.DataFrame(X)
            df['target'] = y
            sns.pairplot(df, hue='target')
            st.pyplot()

        elif plot_type == "Correlation Heatmap":
            st.write("Correlation Heatmap")
            corr_matrix = X.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    def scale_features(X, scaling_method):
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling_method == "RobustScaler":
            scaler = RobustScaler()
        else:
            return X
        return scaler.fit_transform(X)

    X = scale_features(X, scaling_method)

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif clf_name == "SVM":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        elif clf_name == "Logistic Regression":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        elif clf_name == "Decision Tree":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            params["max_depth"] = max_depth
        elif clf_name == "Gradient Boosting":
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0)
            params["n_estimators"] = n_estimators
            params["learning_rate"] = learning_rate
        elif clf_name == "AdaBoost":
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0)
            params["n_estimators"] = n_estimators
            params["learning_rate"] = learning_rate
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "SVM":
            clf = SVC(C=params["C"], probability=True)
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
        elif clf_name == "Logistic Regression":
            clf = LogisticRegression(C=params["C"])
        elif clf_name == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=params["max_depth"])
        elif clf_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"])
        elif clf_name == "AdaBoost":
            clf = AdaBoostClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"])
        elif clf_name == "Naive Bayes":
            clf = GaussianNB()
        return clf

    clf = get_classifier(classifier_name, params)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Classification
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"Classifier = {classifier_name}")
    st.write(f"Accuracy = {acc}")
    st.write(f"F1 Score = {f1}")
    st.write(f"Mean Squared Error = {mse}")
    st.write(f"Mean Absolute Error = {mae}")
    st.write(f"Root Mean Squared Error = {rmse}")
    st.write(f"R² Score = {r2}")

    # Confusion Matrix
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # ROC Curve and AUC (for binary classification)
    if len(np.unique(y)) == 2:
        st.write("ROC Curve and AUC")
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)

    # Feature Importance (for tree-based models)
    if hasattr(clf, 'feature_importances_'):
        st.write("Feature Importance")
        importance = clf.feature_importances_
        feature_names = X.columns  # Adjust if necessary
        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=feature_names, ax=ax)
        st.pyplot(fig)

    # Download Predictions
    if st.sidebar.checkbox("Download Predictions"):
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        pred_csv = pred_df.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV", data=pred_csv, mime='text/csv')

    # Plot PCA if the dataset has more than 1 feature
    if X.shape[1] > 1:
        pca = PCA(2)
        X_projected = pca.fit_transform(X)
        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        st.pyplot(fig)
    else:
        st.warning("Cannot perform PCA with only one feature.")

    # Plot Correlation Matrix
    st.write("Correlation Matrix")
    corr_matrix = np.corrcoef(X.T)
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(X.shape[1]), range(X.shape[1]))
    plt.yticks(range(X.shape[1]), range(X.shape[1]))
    st.pyplot(fig)

