import {useEffect, useState} from 'react'
import './App.css'
import FirstModel from "./components/FirstModel.jsx";

export default function App() {
    const [message, setMessage] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch('http://127.0.0.1:8000/');
            const data = await response.json();
            setMessage(data.message);
        };
        fetchData().then(r => console.log(r));
    }, []);

    return (
        <>
            <div className="flex flex-col mb-20 items-center justify-center">
                {message && <p>{message}</p>}
            </div>
            <div>
                <FirstModel/>
            </div>
        </>
    )
}
