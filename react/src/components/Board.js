import React, { useState, useEffect } from "react";
import { King, Pawn } from "./Pieces";
import * as mathjs from "mathjs";

export default function Board({
  state,
  bindSquares,
  setPieceSelected,
  currentPlayer,
  pieceSelected,
  cardSelected,
  playerData,
}) {
  // 5 x 5 grid of divs
  let i, j;
  let squares = [];
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      squares.push(
        <Square
          key={i + ", " + j}
          pos={[i, j]}
          pieceSelected={pieceSelected}
          bindSquares={bindSquares}
          cardSelected={cardSelected}
          currentPlayer={currentPlayer}
          playerData={playerData}
        />
      );
    }
  }

  // place pieces
  let pieces = [];
  const placePlayer = (data, player) => {
    // king
    pieces.push(
      <King
        key={player + "king"}
        player={player}
        pos={data.king}
        clickSquare={bindSquares}
        setPieceSelected={setPieceSelected}
        currentPlayer={currentPlayer}
        pieceSelected={pieceSelected}
      />
    );
    // pawns
    data.pawns.map((pos, i) =>
      pieces.push(
        <Pawn
          key={player + "pawn" + i}
          pos={pos}
          i={i}
          player={player}
          clickSquare={bindSquares}
          setPieceSelected={setPieceSelected}
          currentPlayer={currentPlayer}
          pieceSelected={pieceSelected}
        />
      )
    );
  };

  // place current player last so we see them on top
  if (currentPlayer === 1) {
    placePlayer(state.player2, 2);
    placePlayer(state.player1, 1);
  }
  else {
    placePlayer(state.player2, 2);
    placePlayer(state.player1, 1);
  }

  return (
    <div>
      <div
        style={{
          width: 600,
          height: 600,
          display: "grid",
          gridTemplateRows: "20% ".repeat(5),
          gridTemplateColumns: "20% ".repeat(5),
        }}
      >
        {squares}
        {pieces}
      </div>
    </div>
  );
}

export function Square({
  pos,
  pieceSelected,
  currentPlayer,
  cardSelected,
  bindSquares,
  playerData,
}) {
  const [i, j] = pos;
  const initColour = (i + j) % 2 === 0 ? "var(--light)" : "transparent";
  const [colour, setColour] = useState(initColour);
  const [validMove, setValidMove] = useState(false);

  const clickHandler = () => {
    if (bindSquares && validMove) bindSquares(pos);
  };

  useEffect(() => {
    const [i, j] = pos;

    function isOccupied() {
      // check king
      if (pos.every((v, i) => v === playerData.king[i])) {
        console.log(playerData.king, playerData.pawns, pos, true);
        return true;
      }
      // otherwise occupied if any pawns are
      return playerData.pawns.some((pawnPos) => {
        return pos.every((v, i) => v === pawnPos[i]);
      });
    }

    const isValidMove = () => {
      let valid = false;
      const cardData = cardSelected.data;
      // where on card is this wrt selected piece
      const [iCard, jCard] = mathjs.subtract(
        mathjs.add([i, j], [2, 2]),
        pieceSelected.pos
      );
      // need to flip if other player
      let iFlipped = currentPlayer === 2 ? 4 - iCard : iCard;
      let jFlipped = currentPlayer === 2 ? 4 - jCard : jCard;
      // card values are 1 and 0
      if (0 <= iFlipped && iFlipped <= 4 && 0 <= jFlipped && jFlipped <= 4) {
        if (cardData[iFlipped][jFlipped] && !isOccupied()) {
          valid = true;
        }
      }
      return valid;
    };

    let newColour = initColour;

    if (pieceSelected && cardSelected) {
      const valid = isValidMove();
      setValidMove(valid);
      if (valid) {
        newColour = "var(--validMove)";
      }
    }
    setColour(newColour);
  }, [
    validMove,
    initColour,
    pieceSelected,
    cardSelected,
    pos,
    currentPlayer,
    playerData,
  ]);

  return (
    <div
      style={{
        background: colour,
        gridRow: pos[0] + 1,
        gridColumn: pos[1] + 1,
      }}
      onClick={clickHandler}
    />
  );
}
