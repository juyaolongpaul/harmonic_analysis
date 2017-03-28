\version "2.10.33"

\header {
  title = "314 - Das alte Jahr vergangen ist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 a8 g8 fis4 b4 
  a8 g8 fis4 g4 b4 
  b4 a8 g8 fis4 b4 
  a8 g8 fis4 g4 b4 
  cis4 dis4 e4 b8 cis8   %5
  d4 cis4 b4 cis4 
  d4 cis4 b4 b4 
  a8 g8 fis4 g4 a4 
  a4 b4 a8 g8 fis4 
  g4 g4 fis4 cis'4   %10
  d4 cis4 b4 fis4 
  b4 b4 ais4 
  
}

alto = \relative c' {
  \partial 4 fis4 
    g4 e4 d4 g4 
  fis4 fis4 e4 gis4 
  e2~ e8 d8 d4 
  c4 c8 b8 b4 g'4 
  g4 fis4 e4 e4   %5
  fis4 fis4 fis4 fis4 
  fis4 fis4. d8 g8 fis8 
  e4 d8 c8 b4 e4 
  d4 d4 cis4 dis4 
  e4 e4. d8 fis4   %10
  fis4 g4 fis4 fis4 
  fis4 eis4 fis4 
  
}

tenor = \relative c' {
  \partial 4 dis4 
    e4 a,4 a4 d4 
  d4 c8 b8 b4 e4 
  e4 a,4 a4 f4 
  e4 fis4 e4 b'4 
  a4 b4 b4 b4   %5
  b4 ais8 e'8 d4 cis4~
  cis8 b4 a8 b4 b4 
  c8 b8 a4 g4 g4 
  fis4 g8 fis8 e4 b'4 
  b4 b4 b4 cis4   %10
  b4 ais4 b8 cis8 d8 cis8 
  b4 cis4 cis4 
  
}

baixo = \relative c' {
  \partial 4 b8  a8 
    g8 e8 cis4 d4 b8 g8 
  d'4 dis4 e4 e8 d8 
  c4 cis4 d4 gis,4 
  a4 dis,4 e4 e'4 
  a,4 a'4 gis4 g4   %5
  fis8 e8 fis4 b,4 ais4 
  b4 fis'4 g8 fis8 e8 d8 
  c4 d4 g,4 cis4 
  d4 g,4 a4 b4 
  e,8 fis8 g8 a8 b4 ais4   %10
  b4 e4 d8 cis8 b8 a8 
  gis4 cis4 fis,4 
  
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