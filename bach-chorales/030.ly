\version "2.10.33"

\header {
  title = "30 - Jesus Christus, unser Heiland"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c' {
  e4 b'4 b4 a4 
  b4 e,8 fis8 g4 g4 
  g4 fis4 e2 
  g4 g4 g4 e4 
  g4 a4 b4 a4   %5
  g4 fis4 e2 
  g4 g4 g4 e4 
  g4 a4 b4 a4 
  g2 r4 d'4 
  e4 d8 c8 b8 a8 g8 a8   %10
  b4 a4 g4 fis4 
  e2.
  
}

alto = \relative c' {
  b4 b4 e8 dis8 e4 
  dis4 e8 dis8 e4 e4 
  e4. dis8 b2 
  e4 e4 d4 c4 
  d4 e8 fis8 g4. fis8~   %5
  fis8 e4 dis8 b2 
  e4 g8 fis8 e4 c4 
  e8 d8 e8 fis8 g4. fis8 
  d2 r4 g8 f8 
  e4 fis4 g8 dis8 e8 d8~   %10
  d8 g4 fis4 e4 dis8 
  b2.
  
}

tenor = \relative c' {
  g4 fis4 g4 fis8 e8 
  fis4 b8 a8 b4 e8 dis8 
  cis4 fis,8 b16 a16 g2 
  b4 c4 g4 g4 
  g8 a16 b16 c4 d8 e16 d16 c16 b16 c8   %5
  b4 b8. a16 g2 
  b8 c8 d4 c4 g4 
  g4 c4 d4~ d8. c16 
  b2 r4 d4 
  c8 b8 a4 g8 c8 b8 a8   %10
  g8 d'8 e8 b8 b4 c8 b16 a16 
  gis2. 
  
}

baixo = \relative c {
  e4. d8 c2 
  b8 a'8 g8 fis8 e8 d8 c8 b8 
  ais4 b4 e,2 
  e'8 d8 c4 b4 c4
  c8 b8 a4 g4 a4   %5
  b2 e2 
  e4 b4 c8 d8 e8 d8 
  c8 b8 a4 g4 d'4 
  g,2 r4 b4 
  c4 d4 e4. fis8   %10
  g8 b,8 cis8 dis8 e8 g,8 a8 b8 
  e,2.
  
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