\version "2.10.33"

\header {
  title = "256 - Jesu, deine tiefen Wunden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    bes4 c4 d4 c4 
    bes4 a4 g4 f4 
    bes4 c4 d4 ees4 
    d4 c4 bes2 }
  d4 ees4 f4 ees4   %5
  d4 c4 d2 
  f4 f4 g4 f4 
  ees4 d4 c2 
  d8 ees8 f4 ees4 d4 
  bes4 c4 d2   %10
  bes2 d4 f4 
  ees4 d4 c4 d4 
  c2 bes2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    f4 f4 f4. e8 
    d4 c8 d8 f8 e8 f4 
    d4 f4 f4 f4 
    f4 f4 f2 }
  bes4 bes4 a4 g8 a8   %5
  bes4 a4 bes2 
  bes4 bes4 bes4 a4 
  g4 f4 f2 
  f4 bes8 aes8 g4 f4 
  g8 fis8 g4. fis16 e16 fis4   %10
  d2 g4 aes4 
  g4 d8 ees8 f4 f4 
  f2 f2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    d4 a4 bes8 a8 g4 
    f8 g8 a8. bes16 c8 g8 a4 
    bes4 a4 bes4 c4 
    c8 bes8 bes8 a8 d2 }
  f4 g4 c,8 d8 ees4   %5
  f4 f4 f2 
  d4 d4 d4 c4 
  bes4 bes4 a2 
  bes8 c8 d4~ d8 c4 bes8~ 
  bes8 a8 g8 a8 bes8 a16 g16 a4   %10
  g2 bes4 bes4 
  bes4 bes4 a4 bes4~ 
  bes8 a16 g16 a4 d2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    bes4 f4 bes,4 c4 
    d8 e8 f4 c'8 c,8 f4 
    g4 f8 ees8 d8 c8 bes8 a8 
    bes4 f4 bes2 }
  bes'8 a8 g4 f4 c4   %5
  d8 ees8 f4 bes,2 
  bes8 c8 d8 bes8 ees8 d8 ees8 f8 
  g8 a8 bes4 f2 
  bes4 bes,4 c4 d4 
  ees4 ees4 d2   %10
  g2 g8 f8 ees8 d8 
  ees8 f8 g8 f8 ees8 d8 c8 bes8 
  f'2 bes,2 
  
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