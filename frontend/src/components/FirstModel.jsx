import  {useState} from "react";

export default function FirstModel() {
    const [file, setFile] = useState(null); // To store the selected file
    const [result, setResult] = useState(""); // To store the result from the server
    const [error, setError] = useState(null); // To handle any errors

    // Handle file selection
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setResult(""); // Clear previous result
        setError(null); // Clear previous error
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
                setResult(`Result: ${data.result}, Confidence: ${data.confidence}`);
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
            <form onSubmit={handleSubmit}>
                <input type="file" accept="image/*" onChange={handleFileChange}/>
                <button type="submit">Upload & Predict</button>
            </form>
            {result && <p style={{color: "green"}}>{result}</p>}
            {error && <p style={{color: "red"}}>{error}</p>}
        </div>
    );
};

