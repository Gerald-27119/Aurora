import {useQuery} from "@tanstack/react-query";
import {getModelsMetrics} from "../http/models.js";
import LoadingIndicator from "./LoadingIndicator.jsx";

export default function Metrics() {
    const {isSuccess, data, isLoading, error} = useQuery({
        queryKey: ['metrics'],
        queryFn: getModelsMetrics
    })

    return (
        <>
            {isSuccess && (
                <div className='flex flex-col items-center mt-6 border-2 border-purple-800 rounded-sm p-4 text-lg text-center'>
                    <table className='table-auto border-collapse border border-gray-800'>
                        <thead>
                        <tr>
                            <th className='border border-gray-600 px-4 py-2'>Model</th>
                            <th className='border border-gray-600 px-4 py-2'>Accuracy</th>
                            <th className='border border-gray-600 px-4 py-2'>Precision</th>
                            <th className='border border-gray-600 px-4 py-2'>Recall</th>
                            <th className='border border-gray-600 px-4 py-2'>F1 Score</th>
                        </tr>
                        </thead>
                        <tbody>
                        {Object.keys(data).map(model => (
                            <tr key={model}>
                                <td className='border border-gray-600 px-4 py-2'>{model}</td>
                                <td className='border border-gray-600 px-4 py-2'>{data[model].Accuracy.toFixed(2)}</td>
                                <td className='border border-gray-600 px-4 py-2'>{data[model].Precision.toFixed(2)}</td>
                                <td className='border border-gray-600 px-4 py-2'>{data[model].Recall.toFixed(2)}</td>
                                <td className='border border-gray-600 px-4 py-2'>{data[model]["F1 Score"].toFixed(2)}</td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            )}
            {isLoading && <LoadingIndicator/>}
            {error &&
                <p className='text-center text-xl text-red-600 border border-yellow-400 rounded-sm'>{error}</p>}
        </>
    )
}