import React, {useState, useEffect} from "react";
import "./App.css";
import Game from "./components/Game";

function App() {
    const [boardState, setBoardState] = useState(null);

    const readApi = () => {
        fetch("http://localhost:5000/getState")
            .then((res) => res.json())
            .then((data) => {
                setBoardState(data);
            });
    };

    const sendMove = (piece, card, pos) => {
        const move = {...piece, id: card.id, pos};
        fetch("http://localhost:5000/sendMove", {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify(move),
        })
            .then((res) => res.json())
            .then((data) => {
                setBoardState(data);
                if (data.winner !== 0) {
                    alert("Winner is player " + data.winner);
                }
            });
    };

    const resetGame = () => {
        fetch("http://localhost:5000/reset")
            .then((res) => res.json())
            .then((data) => {
                setBoardState(data);
            });
    };

    const toggleGameMode = () => {
        fetch("http://localhost:5000/toggleGameMode")
            .then((res) => res.json())
            .then((data) => {
                setBoardState(data);
            });
    };

    const stepBot = () => {
        if (boardState.winner !== 0) {return}
        fetch("http://localhost:5000/stepBot")
            .then((res) => res.json())
            .then((data) => {
                setBoardState(data);
            });
    };

    useEffect(() => {
        if (!boardState) readApi();
    });

    return (
        <div className="App">
            <header className="App-header">
                {boardState ? (
                    <Game
                        state={boardState}
                        readApi={readApi}
                        sendMove={sendMove}
                        resetGame={resetGame}
                        toggleGameMode={toggleGameMode}
                        stepBot={stepBot}
                    />
                ) : (
                    <>No board state, check Flask backend</>
                )}
            </header>
        </div>
    );
}

export default App;
