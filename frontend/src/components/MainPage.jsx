import {useState} from "react";
import {useMutation} from "@tanstack/react-query";
import {predictCarModel} from "../http/models.js";
import LoadingIndicator from "./LoadingIndicator.jsx";
import Metrics from "./Metrics.jsx";

export default function MainPage() {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null)
    const [areMetricsHidden, setAreMetricsHidden] = useState(true)

    const handleClickMetricsBtn = () => {
        setAreMetricsHidden(!areMetricsHidden)
    }

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setResults(null);
        setError(null);
    };

    const {mutate, isPending} = useMutation({
        mutationKey: ['predict'],
        mutationFn: predictCarModel
    })

    const handleSubmit = async (event) => {
        event.preventDefault()

        if (!file) {
            alert("Please upload an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        mutate(formData, {
            onSuccess: (data) => {
                setResults({
                    resnet50: data.resnet50,
                    mobilenetv2: data.mobilenetv2,
                    efficientnetb0: data.efficientnetb0,
                });
            },
            onError: (error) => {
                setError(`Error: ${error.message}`);
            }
        });
    }

    return (
        <div className='bg-gray-950 text-white w-screen h-screen flex flex-col items-center'>
            <p className="mb-10 text-2xl text-center w-1/2 mt-6">
                Twoje zdjęcie samochodu osobowego zostanie przekazane do trzech różnych modeli, które pokażą %
                prawdopodobieństwa jaki to model.
            </p>
            <form onSubmit={handleSubmit} className='border-2 rounded-sm p-4'>
                <input type="file" accept="image/*" onChange={handleFileChange} className='cursor-pointer'/>
                <button type="submit" className='bg-indigo-700 hover:bg-blue-700 p-3 rounded-md'>Upload & Predict
                </button>
            </form>
            {isPending && <LoadingIndicator/>}
            {results && (
                <div className='border-2 border-purple-800 rounded-sm p-4 mt-6 text-lg flex flex-col space-y-4'>
                    <div className='flex'>
                        <h3>ResNet50 Prediction:</h3>
                        <p>
                            Result: {results.resnet50.result}, Confidence: {results.resnet50.confidence}
                        </p>
                    </div>
                    <div className='flex'>
                        <h3>MobileNetV2 Prediction:</h3>
                        <p>
                            Result: {results.mobilenetv2.result}, Confidence: {results.mobilenetv2.confidence}
                        </p>
                    </div>
                    <div className='flex'>
                        <h3>EfficientNetB0 Prediction:</h3>
                        <p>
                            Result: {results.efficientnetb0.result}, Confidence: {results.efficientnetb0.confidence}
                        </p>
                    </div>
                </div>
            )}
            {error &&
                <p className='text-center text-xl text-red-600 border border-yellow-400 rounded-sm'>{error}</p>}
            <button onClick={handleClickMetricsBtn} className='text-center text-lg bg-green-700 hover:bg-emerald-700 px-2 py-1 rounded-md mt-6'>
                {areMetricsHidden ? 'Show models metrics' : 'Hide models metrics'}</button>
            {!areMetricsHidden && <Metrics />}
        </div>
    );
}
