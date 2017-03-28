\header {
  objetivo = "testar repetição"
}

\relative c' {
  \time 3/4
  \repeat volta 2 {
    <c e g>2. <b d g> <c e g>
  }
  \alternative {
    {<c e a>}
    {<b d g>}
  }
  \bar "|."
}