\version "2.10.33"

\header {
  title = "56 - Christum wir sollen loben schon"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \partial 4 e8  fis8 
    g4 g4 a4 b4~ 
  b4 e,4 a4 g4 
  fis2. a4 
  b8 cis8 d2 cis4 
  b4 a4 b8 cis8 d4   %5
  d2. a4 
  b8 cis8 d2 cis4 
  b4 a4 b8 cis8 d4~ 
  d4 a8 g8 fis8 g8 a4 
  b2. e,8 fis8   %10
  g4 g8 a8 b8 c8 d4~ 
  d4 b4 a8 g8 fis8 g8 
  a4 g4 fis2~ 
  fis1~ 
  fis1~   %15
  fis1 
  
}

alto = \relative c' {
  \partial 4 b4 
    e4 e4 e4 fis4~ 
  fis4 e4 dis4 e4 
  dis2. d4 
  g4 fis8 g8 a4 e8 fis8 
  g4 fis8 e8 d4 g4   %5
  fis2. fis8 e8 
  d8 e8 fis2 fis8 e8 
  d8 e8 fis4 g4 d8 e8 
  fis2. e4 
  fis2. e8 dis8   %10
  e8 fis8 g4 fis8 e8 d8 c8 
  b8 c8 d8 e8 fis4. e8 
  dis4 e2 d8 cis8 
  d8 e8 fis2 gis8 ais8 
  b1~   %15
  b4 ais8 gis8 ais2 
  
}

tenor = \relative c' {
  \partial 4 g8  a8 
    b4 b4 a2 
  g8 a8 b4 a4 b4 
  b2. a4 
  e'4 a,4. b8 cis8 d8 
  e8 d8 cis4 b4 b4   %5
  a2. d8 e8 
  fis4 b,2 a4 
  b8 cis8 d4 g,8 a8 b4~ 
  b8 cis8 d4 cis8 dis8 e4 
  dis2. b4   %10
  b4 b8 c8 d8 e8 fis4 
  g4 g,4 d'8 e8 fis4~ 
  fis4 e8 d8 cis4 fis,4 
  b8 fis8 b4. ais8 b8 cis8 
  d1~   %15
  d4 cis8 b8 cis2 
  
}

baixo = \relative c {
  \partial 4 e4 
    e4 e8 d8 cis4 dis4 
  e8 fis8 g4 fis4 e4 
  b2. fis'4 
  e4 d8 e8 fis8 g8 a4 
  e4 fis4 g8 a8 b8 cis8   %5
  d2. d,8 cis8 
  b4. cis8 d8 e8 fis4 
  g4 fis4 e4 b8 cis8 
  d8 e8 fis8 g8 a8 b8 c4 
  b2. g8 fis8   %10
  e4 e4 d8 c8 b8 a8 
  g8 a8 b8 c8 d4. c8 
  b4 c8 b8 ais2 
  b8 cis8 d8 e8 fis4. e8 
  d8 cis8 b8 ais8 b8 d8 cis8 b8   %15
  fis'1 
  
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