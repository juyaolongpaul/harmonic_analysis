\version "2.10.33"

\header {
  title = "227 - Lobet den Herren, denn er ist sehr freundlich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    d4 g4 g2 
  a2 r4 a4 
  bes8 c8 d4 cis2 
  d2 bes4 bes4 
  c4 c4 d2   %5
  bes2 r4 d4 
  c4 c4 c4 a4 
  d4 c4 bes4 a4 
  g2 a2 
  bes4 a4 g4 fis4   %10
  g2 a2 
  r4 d4 bes4 bes4 
  bes4 d4 c4 bes4 
  a4 g4 a2 
  g2 r4 g4   %15
  d4 g4 g2 
  a2 r4 a4 
  bes8 c8 d4 cis2 
  d2.
  
}

alto = \relative c' {
  \partial 4 d8  c8 
    bes8 c8 d4. c16 bes16 a8 g8 
  d'2 r4 fis4 
  g4 a4 a2 
  a2 g8 a8 bes4 
  bes4 a4 bes4. aes8   %5
  g2 r4 bes4 
  a8 g8 f8 e8 f8 g8 a8 g8 
  a8 fis8 g8 a4 g4 fis8 
  g8. fis16 g8 g,8 d'2 
  d8 e8 f8 fis8 g8 ees8 d4~   %10
  d8 c8 d8 bes8 d2 
  r4 fis4 d4 g4 
  f4. g16 f16 e8 fis8 g4~ 
  g8 fis8 g4 g4 fis4 
  d2 r4 d4   %15
  d4 d4. c16 bes16 a8 g8 
  d'2 r4 fis4 
  g4 a8 e8 e8 fis8 g4 
  fis2. 
  
}

tenor = \relative c' {
  \partial 4 bes8  a8 
    g8 a8 bes4. a16 g16 c4 
  fis,2 r4 d'4 
  d8 e8 f4 e8 d8 e16 fis16 g8 
  fis2 d4 f4 
  g4 f4 f4 f4   %5
  f2 r4 f4 
  f8 bes,8 a8 g8 a8 bes8 c4~ 
  c8 bes4 a8 bes16 c16 d8 d8. c16 
  bes8 d4 c8 fis,2 
  g4 c4 d8 c16 bes16 a4   %10
  g8 d8 g2 fis4 
  r4 a4 g8 a8 bes8 c8 
  d8 ees8 d8 bes8 g8 a8 bes16 c16 d8 
  ees8 d16 c16 bes8 c16 d16 ees4 d8. c16 
  bes2 r4 bes8 a8   %15
  g8 a8 bes8 a8 g4. c8 
  fis,2 r4 d'4 
  d4 a8 bes8 a2 
  a2.
  
}

baixo = \relative c {
  \partial 4 g8  a8 
    bes8 a8 g4 ees'2 
  d2 r4 d4 
  g4 f8 g8 a2 
  d,2 g8 f8 ees8 d8 
  ees8 c8 f8 f,8 bes8 c8 d8 bes8   %5
  ees2 r4 bes4 
  f'8 g8 a8 bes8 a8 g8 f8 e8 
  fis8 d8 e8 fis8 g4 d4 
  ees2 d2 
  g,4 a4 bes8 c8 d8 c8   %10
  bes8 a8 bes8 g8 d'2 
  r4 d4 g8 f8 ees4 
  d8 c8 bes4. a8 g8 bes8 
  c8 d8 ees8 d8 c8 a8 d4 
  g,2 r4 g8 a8   %15
  bes8 a8 g8 f8 ees2 
  d2 r4 d'4 
  g4 f8 g8 a4 a,4 
  d2.
  
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