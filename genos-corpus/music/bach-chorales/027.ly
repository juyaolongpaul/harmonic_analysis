\version "2.10.33"

\header {
  title = "27 - Es spricht der Unweisen Mund"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes4 a8 g8 f4 bes4 
    c4 d4 bes4 f4 
    bes4 c4 d4 ees4 
    c8 bes8 c4 bes4 } bes4 
  g4 a4 bes8 a8 g4   %5
  f4 e4 f4 f4 
  bes4 c4 d4 ees4 
  d4 c4 d4 c4 
  d4 ees4 f8 ees8 d4 
  c8 bes8 c4 bes   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f4 ees4 d4 d4 
    ees4 f4 ees4 c4 
    f4 f4 f4 g4 
    f2 f4 } f4 
  e4 f2 e8 d8   %5
  c4 c4 c4 c4 
  f8 g8 a4 bes4 c4~ 
  c8 bes4 a8 bes4 f4 
  f4 g4 a4 f4 
  f2 f4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 c4. bes8 bes4~ 
    bes8 a8 bes8 aes8 g4 a4 
    bes4 ees4 d8 c8 bes4~ 
    bes8 a16 g16 a4 d4 } d4 
  c4 c4 d4 g,8 a16 bes16   %5
  c8 bes16 a16 g8 a16 bes16 a4 a4 
  bes4 ees4 f4 g8 f8 
  f4 f4 f4 f8 ees8 
  d8 c8 bes4 c4. bes8~ 
  bes8 a16 g16 a4 d   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 bes4 
    bes4 c4 d4 g4 
    f4 bes,4 ees4 ees4 
    d4 a4 bes4 ees4 
    f2 bes,4 } bes4~ 
  bes4 a4 g4 c8 bes8   %5
  a8 bes8 c4 f,4 f'8 ees8 
  d8 ees8 d8 c8 bes8 bes'4 a8 
  bes4 f4 bes,4 a'4 
  bes8 a8 g4 f4 bes4 
  f2 bes,4   %10
  
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