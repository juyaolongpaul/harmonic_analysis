\version "2.12.0"
\header {
  title = "Kostka Exemplo 7-1"
}

\new PianoStaff <<
  \new Staff {
    \time 3/4
    \key g \major
    \relative c' {
      <e g c>2
      <e g b>4
      <fis a d>
      <g c e>
      <a c e>
      <g b e>
      <g c e>
      <a c fis>
      <a c e>2.
      \bar "|."
    }
  }
  \new Staff  {
    \time 3/4
    \key g \major
    \clef bass
    \relative c{
      c2 e4 
      d c a 
      e' c a 
      a2.
      \bar "|."
    }
  }
>>