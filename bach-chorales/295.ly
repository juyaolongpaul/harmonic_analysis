\version "2.10.33"

\header {
  title = "295 - Herr Jesu Christ, mein's Lebens Licht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 a4 b4 g4 
  fis8 g16 a16 fis4 e4 e8 fis8 
  g4 a4 b4 a4 
  d4 cis4 b4 b4 
  e4 fis4 e4 b4   %5
  cis4 b4 a4 a8 b8 
  cis4 dis4 e4 d4 
  cis8 d16 e16 cis4 b4 
  
}

alto = \relative c' {
  \partial 4 dis4 
    e4 e4 fis4 e4 
  e4 dis4 b4 b4 
  e4 d4 d4 d8 e8 
  fis4 fis8. e16 d4 e4 
  e4 d4 e4 eis4   %5
  fis4 fis8 eis8 fis4 fis8 gis8 
  a4 a4 b8 cis cis b8 
  b8 ais16 gis16 ais4 fis4 
  
}

tenor = \relative c {
  \partial 4 fis4 
    g4 a8 g8 fis4 b4 
  c4 b8 a8 g4 g8 a8 
  b4 a4 g4 fis4 
  b4 b8 ais8 fis4 g4 
  a4 a8 b8 cis4 d4   %5
  cis4 gis8 cis8 cis4 d4 
  e4 fis4 e8 cis8 d8 fis8 
  g8 cis,8 fis8 e8 dis4 
  
}

baixo = \relative c {
  \partial 4 b4 
    e4 c'4 dis,4 e4 
  a,4 b4 e4 e4 
  e4 fis4 g4 d8 cis8 
  b4 fis'4 b,4 e8 d8 
  cis4 d4 a'4 gis4   %5
  a8 b8 cis4 fis,4 d4 
  a'4 gis8 fis8 gis8 ais8 b4 
  e,4 fis4 b,4 
  
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