\version "2.10.33"

\header {
  title = "46 - Vom Himmel hoch, da komm' ich her"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \partial 4 d4 
    cis4 b4 cis4 a4 
  b4 cis4 d4 d4 
  d4 a4 a4 fis8 g8 
  a4 g4 fis4 fis4 
  b4 b4 a8 b8 cis4   %5
  d8 cis8 b4 a4 d4 
  cis4 b4 a4 b8 a8 
  g8 fis8 e4 d2 
  
}

alto = \relative c'' {
  \partial 4 a4 
    a8 gis8 fis4 e4 fis8 e8 
  d4 e4 fis4 fis8 g8 
  a8 g8 fis4 e4 d4 
  d4 e4 d4 d8 e8 
  fis4 e4 e8 gis8 a4   %5
  a4. gis8 e4 fis8 e8 
  e8 fis4 e8 cis8 d8 b8 cis8 
  d4. cis8 a2 
  
}

tenor = \relative c' {
  \partial 4 fis4 
    e4 d4 cis4 d8 cis8 
  b8 a8 g4 a4 a4 
  d4 d4 cis4 c8 b8 
  a4 a4 a4 a4 
  a4 gis4 a8 d8 cis8 b8   %5
  a8 cis8 fis8 e8 cis4 d8 b8~ 
  b8 a8 a8 g8 a4 g8 fis8 
  d4 a'8 g8 fis2 
  
}

baixo = \relative c {
  \partial 4 d4 
    a'4. gis8 a8 g8 fis4 
  g8 fis8 e4 d4 d8 e8 
  fis4 fis8 g8 a4 a8 g8 
  fis8 e8 d8 cis8 d4 d4 
  dis4 e8 d8 cis8 b8 a4   %5
  fis'8 e8 d8 e8 a,4 b'8 e,8 
  a4 d,8 g4 fis8 g8 a8 
  b8 g8 a8 a,8 d2 
  
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