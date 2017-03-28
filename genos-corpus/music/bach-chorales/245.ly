\version "2.10.33"

\header {
  title = "245 - Christe, der du bist Tag und Licht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 g4 
  bes4 g4 f4 g8 a8 
  bes4 a4 g4 bes4 
  bes4 bes4 bes4 f4 
  g4 bes4 a4 a4 
  c4 c4 c8 bes16 a16 g8 a8   %5
  bes4 a4 g4 a4 
  bes4 g4 f4 g8 a8 
  bes4 a g
  
}

alto = \relative c' {
  \partial 4 d4 
  d4 d4 d4 d8 fis8 
  g4. fis8 d4 g4 
  g4 g4 f8 ees4 d16 c16 
  d8 e16 f16 g4 f4 f4 
  f4 e8 fis8 g8 fis8 g8 d8   %5
  d8 g4 fis8 d4 fis4 
  g4 d4 d4 d8 fis8 
  g4. fis8 d4
  
}

tenor = \relative c' {
  \partial 4 d8  c8 
  bes4 bes4 a4 bes8 c8 
  d4 ees8 d16 c16 bes4 d4 
  d4 d4 d8 g,8 a8 bes8~ 
  bes8 a8 g8 c8 c4 c8 bes8 
  a8 g8 a4 g8 c8 bes8 a8   %5
  g8 d'8 ees8 d16 c16 bes4 d4 
  d4 bes4 a4 bes8 c8 
  d4 ees8 d16 c16 b4
  
}

baixo = \relative c {
  \partial 4 bes8  a8 
  g8 a8 bes8 c8 d8 c8 bes8 a8 
  g8 bes8 c8 d8 g,4 g'4 
  g,8 a8 bes8 c8 d8 ees8 f4 
  f4 e4 f4 f,8 g8 
  a8 bes8 c8 d8 ees4. fis,8   %5
  g8 bes8 c8 d8 g,4 d'4 
  g,8 a8 bes8 c8 d8 c8 bes8 a8 
  g8 bes8 c8 d8 g,4
  
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