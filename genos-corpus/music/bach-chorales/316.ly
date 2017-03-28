\version "2.10.33"

\header {
  title = "316 - Christus, der ist mein Leben"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
  b2 a4 
  b2 c4 
  d2. 
  b2 r4 
  e2.~   %5
  e2. 
  d2.~ 
  d2. 
  c2.~ 
  c8 b8 c4 r4   %10
  b4 e8 c8 a4 
  b2 d4 
  e2 fis4 
  g2 fis4 
  e2.   %15
  d2 b4 
  a2 g4 
  g2 fis4 
  g2
  
}

alto = \relative c' {
  \partial 4 d4 
  g2 fis4 
  g2 g4 
  g4. fis16 e16 fis4 
  d2 r4~ 
  r2.   %5
  f2.~ 
  f2.~ 
  f2. 
  e2. 
  fis4 fis4 r4   %10
  g8 fis8 e4 fis8 e8 
  dis2 g4 
  g2 a4 
  b2 a4 
  b8 a8 g4. a16 g16   %15
  fis2 g4 
  g4 fis4 e4 
  e4 d2 
  d2
  
}

tenor = \relative c' {
  \partial 4 b4 
  d2 d4 
  d2 c4 
  b4. a16 g16 a4 
  g2 r4~ 
  r2.~   %5
  r2. 
  gis2.~ 
  gis2. 
  a2.~ 
  a4 a4 r4   %10
  g8 b8 b8 a16 b16 c4 
  fis,2 d'4 
  c2 c4 
  d2 d4 
  d4. cis16 b16 cis4   %15
  a2 e'4 
  e4 b4 b4 
  c8 b8 a8 b8 c4 
  b2 
  
}

baixo = \relative c' {
  \partial 4 g4 
  g2 d4 
  g4 fis4 e4 
  b8 c8 d4 d4 
  g,2 r4~ 
  r2.~   %5
  r2.~ 
  r2. 
  c2.~ 
  c4. b8 c8 a8 
  d4 d4 r4   %10
  e8 d8 c4 c4 
  b2 b'4 
  b2 a4 
  g4 b4 d4 
  g,8 fis8 e4 a4   %15
  d,2 e8 d8 
  cis4 dis4 e4 
  c4 d2 
  g,2
  
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