\version "2.10.33"

\header {
  title = "208 - Als vierzig Tag' nach Ostern"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key e \minor
}


soprano = \relative c' {
  e4 e4 e4 
  b'2 b4 
  a4 b4 g8 a8 
  b4 a4 a4 
  b2 g4   %5
  fis4 g4 a4 
  b4 a4 g4 
  g2 fis4 
  g2 a4 
  b2 b4   %10
  a2 a4 
  g2 g4 
  fis2. 
  e4 fis4 dis4 
  e2 e4   %15
  a2 a4 
  b2 b4 
  g2 g4 
  fis2. 
  e8 fis8 g8 a8 b4   %20
  b2 a4 
  b2. 
  
}

alto = \relative c' {
  b4 b4 b4 
  fis'2 g4~ 
  g4 fis4 e4 
  fis4 e8 g8 fis8 e8 
  fis2 b,8 cis8   %5
  d2 e8 fis8 
  g4. fis4 e8 
  e8 d8 c4 d4 
  d2 fis4 
  g2 g4   %10
  fis8 e8 fis4 dis4 
  e8 dis8 e2~ 
  e4 dis2 
  e4 c4 b4 
  b2 b4   %15
  e4 d4 c4 
  d4 e4 fis4~ 
  fis4 e8 dis8 e4 
  e4 dis2 
  b4 e4 fis4   %20
  e4. g8 fis8 e8 
  dis2. 
  
}

tenor = \relative c' {
  g4 g4 g4 
  fis4 dis'4 e4 
  e4 b4 b4 
  b4 c8 e8 dis8 e8 
  dis2 e4   %5
  a,4 b4 c4 
  d4. c8 b4 
  c4. b8 a4 
  b2 d4 
  d2 e8 d8   %10
  c4 b4 b4 
  b2 b4 
  b2. 
  g4 a4 fis4 
  g2 g4~   %15
  g4 fis4 fis4 
  g4 b2 
  b2 b4 
  b2. 
  g4 b4 b8 a8   %20
  g8 fis8 e8 dis8 e4 
  fis2. 
  
}

baixo = \relative c {
  e4 g4 e4 
  dis4 b4 e4 
  cis4 dis4 e4 
  d4 c8 b8 c4 
  b2 e4   %5
  d8 c8 b4 a4 
  g4 d'4 e4 
  c4 a4 d4 
  g,2 d'4 
  g4 fis4 e4~   %10
  e4 dis4 b4 
  e4 g4 e4 
  b2. 
  c4 a4 b4 
  e2 e4   %15
  cis4 d8 c8 b8 a8 
  g4 g'8 fis8 e8 dis8 
  e4 e,8 fis8 g8 a8 
  b2. 
  e,4 e'4 dis4   %20
  e8 d8 c8 b8 c4 
  b2. 
  
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