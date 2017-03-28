\version "2.10.33"

\header {
  title = "284 - Herr Jesu Christ, wahr'r Mensch und Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  \partial 4 f4 
    f4 f4 e4 c4 
  d4 e4 f4 f4 
  e4 d4 c4 bes4 
  a4 g4 a4 a4 
  g4 a8 b8 c4 d4   %5
  c4 b4 c4 c4 
  d4 e4 f4 e4 
  d4 d4 c4 c4 
  bes4 a4 d4 c4 
  bes4 a4 g4 g4   %10
  a4 b4 c4 d4 
  c4 b4 c4 
  
}

alto = \relative c'' {
  \partial 4 a4 
    a4 a4 g4 a8 g8 
  f4 g4 a4 a4 
  g8 f8 f8 g8 g8 a8 d,8 e8 
  f4 e8 d8 cis4 d4 
  d8 c8 c4 c8 f8 f4   %5
  f8 e8 d4 e4 f4 
  f4 bes4 a8 g8 g4 
  fis4 gis4 a4 a4 
  a8 g8 g8 f8 f8 ees8 ees8 d8 
  d8 e8 f4 e4 e4   %10
  ees8 d8 d4 ees4 f4 
  f8 ees8 d4 e4 
  
}

tenor = \relative c' {
  \partial 4 c4 
    c4 c8 d8 e4 f8 e8 
  d8 c8 bes4 c4 d8 c8 
  bes8 c8 d4 e8 d8 d8 cis8 
  d8 a8 bes4 e,4 f4 
  g4 f4 e8 a8 a8 g8   %5
  g4 g4 g4 a4 
  bes4 bes4 c8 d8 e4 
  a,8 b16 c16 d8 e8 e4 e4 
  f8 c8 c4 bes4 fis4 
  g4 c4 c4 c4   %10
  c4 g4 g4 aes8 g8 
  g4 g4 g4 
  
}

baixo = \relative c {
  \partial 4 f8  g8 
    a4 g8 f8 c'8 bes8 a4 
  bes8 a8 g4 f4 d4 
  g8 a8 bes4 e,8 fis8 g4 
  d8 c8 bes4 a4 d4 
  e4 f8 g8 a4 bes4   %5
  c4 g4 c,4 f4 
  bes4 a8 g8 a8 b8 c4 
  c8 b16 a16 b8 e,8 a4 a,4 
  d8 e8 f4 bes,8 c8 d4 
  g,4 a8 bes8 c4 c'4   %10
  fis,4 f4 ees4 b4 
  c4 g4 c4 
  
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