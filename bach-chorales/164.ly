\version "2.10.33"

\header {
  title = "164 - Herr Gott, dich loben alle wir"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 bes4 
    bes2 a4 
  g2 f4 
  bes4 c2 
  d2 d4 
  d2 d4   %5
  c2 d4 
  ees4 d2 
  c2 bes4 
  c2 d4 
  c2 bes4   %10
  g4 a2 
  bes2 f'4 
  d2 bes4 
  c2 ees4 
  d4 c2   %15
  bes2 
  
}

alto = \relative c' {
  \partial 4 f4 
    g2 f4 
  ees2 d4 
  d4 g4 f4 
  f2 a4 
  g2 g4   %5
  g4 f4 f4 
  f4 f2 
  f2 g4 
  f2 f8 bes8 
  g4 f8 ees8 d4   %10
  g4 g4 fis4 
  g2 f4 
  f2 g4 
  f2 f4 
  f4 g4 f8 ees8   %15
  d2 
  
}

tenor = \relative c' {
  \partial 4 d4 
    d2 d4 
  bes2 bes4 
  bes2 a4 
  bes2 a4 
  bes2 bes4   %5
  bes4 a4 bes4 
  c4 bes2 
  a2 d4 
  c2 bes4 
  bes4 a4 bes4   %10
  bes8 d8 ees4 d4 
  d2 c4 
  d2 d4 
  a2 a4 
  bes2 a4   %15
  f2 
  
}

baixo = \relative c {
  \partial 4 bes8  a8 
    g8 a8 bes8 c8 d8 bes8 
  ees8 f8 g8 a8 bes8 a8 
  g8 f8 ees8 d8 ees8 f8 
  bes,2 fis4 
  g8 d'8 g8 f8 ees8 d8   %5
  ees8 c8 f8 ees8 d8 c8 
  bes8 a8 bes8 c8 d8 ees8 
  f2 g4 
  a8 f8 g8 a8 bes8 g8 
  ees8 d8 ees8 f8 g8 d8   %10
  ees8 bes8 c8 a8 d8 d,8 
  g2 a'4 
  bes8 c8 bes8 a8 g8 f8 
  ees8 g8 f8 ees8 d8 c8 
  bes8 d8 ees8 c8 f8 f,8   %15
  bes2 
  
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