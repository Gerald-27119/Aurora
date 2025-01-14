export default function LoadingIndicator() {
    return (
        <div className='flex justify-center'>
            <div className="lds-ring">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
        </div>
    );
}