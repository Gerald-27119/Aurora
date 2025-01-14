import axios from "axios";

export async function predictCarModel(image) {
    return (await axios.post('http://localhost:8000/predict', image)).data
}

export async function getModelsMetrics() {
    return (await axios.get('http://localhost:8000/metrics')).data
}