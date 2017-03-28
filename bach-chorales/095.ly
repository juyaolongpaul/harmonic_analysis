\version "2.10.33"

\header {
  title = "95 - Werde munter, mein Gemüte"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    d4 ees4 f4 f4 
    ees4 d4 c4 c4 
    d4 ees4 f4 ees8 d8 
    c4. bes8 bes2 }
  c4 d4 ees4 ees4   %5
  d4 d4 c2 
  ees4 f4 g4 g4 
  f4 f4 ees2 
  d4 ees4 f4 f4 
  ees4 d4 c4 c4   %10
  d4 ees4 f4 ees8 d8 
  c4. bes8 bes4 bes4
}

alto = \relative c' {
  \repeat volta 2 {
    f4 g8 a8 bes4 bes4 
    bes8 a8 bes4 a4 a4 
    bes4 a8 g8 f4 g4 
    g4 f4 f2 }
  f4 f4 ees8 f8 g4   %5
  aes4 g8 f8 ees2 
  g4 bes4 bes8 aes8 bes8 g8 
  c4 bes8 aes8 g2 
  g4 g4 f8 ees8 d4 
  ees4 f8 g8 a4 a4   %10
  bes4 a8 g8 f4 g4 
  g4 f4 f4 f4
}

tenor = \relative c' {
  \repeat volta 2 {
    bes4 bes4 bes8 c8 d4 
    ees4 f4 f4 f4 
    f4 c4 c8 a8 bes4 
    bes4 a4 d2 }
  a4 b4 c4 c4   %5
  c4 b4 g2 
  c4 d4 ees4 ees4 
  ees4 d4 bes2 
  bes4 bes4 bes4 bes4 
  bes8 a8 bes4 f'4 f4   %10
  f4 ees4 d8 c8 bes4 
  bes4 a8 ees'8 d4 d4
}

baixo = \relative c' {
  \repeat volta 2 {
    bes8 a8 g4 d4 g4 
    c,4 d8 ees8 f4 f4 
    bes,4 c4 d4 g8 f8 
    ees4 f4 bes,2 }
  f'8 ees8 d4 c8 d8 ees8 c8   %5
  f4 g4 c,2 
  c'4 bes8 aes8 g8 f8 ees4 
  aes4 bes4 ees,2 
  g4 f8 ees8 d4 g4 
  c,4 d8 ees8 f4 f4   %10
  bes,4 c4 d4 g8 f8 
  ees4 f4 bes,4 bes4
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}