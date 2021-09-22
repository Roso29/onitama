import React from "react";

function PawnSvg({player}) {
    return <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <polygon points="0,100 50,0, 100,100" fill={player === 1 ? "var(--lightPiece)" : "var(--darkPiece)"}/>
    </svg>;
}

function KingSvg({player}) {
    return <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <polygon points="0,100 16,0 32,60 48,0 64,60 80,0 100,100" fill={player === 1 ? "var(--lightPiece)" : "var(--darkPiece)"}/>
    </svg>;
}

function Piece({
                   name,
                   pos,
                   player,
                   i,
                   currentPlayer,
                   clickSquare,
                   setPieceSelected,
               }) {
    let [row, col] = pos;
    const clickHandler = () => {
        if (currentPlayer === player) {
            console.log("Piece clicked", name, pos, currentPlayer, i, player);
            setPieceSelected({name, player, i, pos});
        } else if (player !== currentPlayer) {
            // try to take
            clickSquare(pos);
        }
    };
    return (
        <div
            onClick={clickHandler}
            style={{
                gridRow: row + 1,
                gridColumn: col + 1,
                height: "80%",
                width: "80%",
                overflow: "hidden",
                position: "relative",
                alignSelf: "end",
                justifySelf: player === 1 ? "start" : "end"
            }}
        >
            {name === "king" ? <KingSvg player={player}/> : <PawnSvg player={player}/>}
            {(i || i === 0) ? <p style={{position: "absolute", top: 0, right: 0, fontWeight: 600,}}>{i}</p> : null}
        </div>
    );
}

function King(props) {
    return <Piece {...props} name={"king"}/>;
}

function Pawn(props) {
    return <Piece {...props} name={"pawn"}/>;
}

export {King, Pawn};
