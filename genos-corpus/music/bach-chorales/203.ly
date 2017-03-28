\version "2.10.33"

\header {
  title = "203 - O Mensch, schau Jesum Christum an"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \minor
}


soprano = \relative c'' {
  g4 g4 g4 
  g2 ees4 
  d2 d4 
  a'2 bes4 
  a2 g4   %5
  f2 e4 
  d2 d4 
  g2 g4 
  fis2 fis4 
  g2 g4   %10
  a2 a4 
  a2 a4 
  bes2 bes4 
  a2. 
  d2 c4   %15
  bes2 a4 
  g4 a4 b4 
  c2 a4 
  bes2 a4 
  g2 fis4   %20
  g2. 
  
}

alto = \relative c' {
  d8 c8 d4 b4 
  c4 b4 c8 a8 
  d8 c8 d4 d4 
  d2 d4 
  cis4 d4 e4   %5
  e8 cis8 d4. cis8 
  a2 a4 
  d2 d4 
  d4 ees4 d4 
  d4 c8 bes8 c4   %10
  fis,2 fis'4 
  fis2 fis4 
  g2 g4 
  g4 fis8 e8 fis4 
  fis8 g8 a2~   %15
  a4 g4 d4 
  g4 f2 
  g2 fis4 
  g4 f4 ees4 
  ees4. ees8 d4   %20
  d2. 
  
}

tenor = \relative c' {
  bes8 a8 g4. f8 
  ees8 f8 g4 fis4 
  g8 a8 bes4 bes4 
  a2 g4 
  a2 a4   %5
  a4 bes8 a8 g8 a8 
  fis2 fis4 
  g8 a8 bes2 
  a8 bes8 c4 a4 
  g4. bes8 a8 g8   %10
  d'2 d4 
  d2 d4 
  d2 d4 
  d2. 
  a4 d2   %15
  d2 d4~ 
  d4 c4 d4 
  ees2 d4 
  d2 c4 
  c8 bes8 a4. c8   %20
  b2. 
  
}

baixo = \relative c {
  g8 a8 bes4 g4 
  c8 d8 ees4 c4 
  bes8 a8 g4 g'4 
  fis2 g4~ 
  g4 f8 e8 d8 cis8   %5
  d4 g,4 a4 
  d,2 d'8 c8 
  bes4 g8 a8 bes8 c8 
  d4 a8 bes8 c8 a8 
  bes4 ees8 d8 ees4   %10
  d2 d,4 
  d'8 ees8 d8 c8 bes8 a8 
  g8 fis8 g8 a8 bes8 c8 
  d2. 
  d8 e8 fis8 e8 fis8 d8   %15
  g8 fis8 g8 a8 g8 f8 
  ees8 d8 ees8 f8 ees8 d8 
  c2 d4 
  g8 g,8 a8 bes8 c8 d8 
  ees8 d8 c4 d4   %20
  g,2. 
  
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