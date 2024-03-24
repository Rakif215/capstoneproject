function toggleAnswer(questionNumber) {
  var answerId = "answer" + questionNumber;
  var answerElement = document.getElementById(answerId);

  if (answerElement.style.display === "none") {
      answerElement.style.display = "block";
  } else {
      answerElement.style.display = "none";
  }
}


function formatScore(score) {
  return parseFloat(score).toFixed(2);
}