\version "2.12.0"
\header {
  title = "Kostka Exemplo 7-1"
}

\new PianoStaff <<
  \new Staff {
    \time 3/4
    \key g \major
    \relative c' {
      <d g b>2
      <d a' d>4
      <d g b>
      <c fis a>
      <b d g>
      <g c e>
      <c e a>
      <c d fis>
      <b d g>2.
      \bar "|."
    }
  }
  \new Staff  {
    \time 3/4
    \key g \major
    \clef bass
    \relative c{
      g2 fis4
      g a b
      c a d
      g,2.
      \bar "|."
    }
  }
>>
