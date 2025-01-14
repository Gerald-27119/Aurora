import { useState } from "react";

export default function FirstModel() {
    const [file, setFile] = useState(null); // To store the selected file
    const [results, setResults] = useState(null); // To store the results from both models
    const [error, setError] = useState(null); // To handle any errors

    // Handle file selection
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setResults(null); // Clear previous results
        setError(null); // Clear previous errors
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) {
            alert("Please upload an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setResults({
                    resnet50: data.resnet50,
                    mobilenetv2: data.mobilenetv2,
                });
            } else {
                const errorData = await response.json();
                setError(`Error: ${errorData.error}`);
            }
        } catch (err) {
            setError(`Error: ${err.message}`);
        }
    };

    return (
        <div>
            <p className="mb-10">
                # The model is better in distinguishing cars, trucks, vans from tanks than random objects from tanks (theoretically, because it was trained on tanks and other vehicles)
            </p>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <button type="submit">Upload & Predict</button>
            </form>
            {results && (
                <div style={{ marginTop: "20px" }}>
                    <h3>ResNet50 Prediction:</h3>
                    <p style={{ color: "green" }}>
                        Result: {results.resnet50.result}, Confidence: {results.resnet50.confidence}
                    </p>
                    <h3>MobileNetV2 Prediction:</h3>
                    <p style={{ color: "orange" }}>
                        Result: {results.mobilenetv2.result}, Confidence: {results.mobilenetv2.confidence}
                    </p>
                </div>
            )}
            {error && <p style={{ color: "red" }}>{error}</p>}
        </div>
    );
}
