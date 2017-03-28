\version "2.10.33"

\header {
  title = "209 - Dir, dir, Jehova, will ich singen"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key bes \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    bes4 d8 c8 d8 bes8 
    g'4 ees2 
    f4 bes,4 ees4 
    c2 bes4 
    c4 ees4 d4     %5
    c4 d8 c8 d8 bes8 
    a8 bes8 bes4. c8 
    c2 } a4 
  bes4 c4 d4 
  g,4 g'8 ees8 c8 b8   %10
  c8 d8 d4. ees8 
  ees2 c4 
  a4 bes8 a8 g8 f8 
  f'4. bes,8 g'4~ 
  g8 f8 ees8 d c8 bes8   %15
  bes2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    f4 bes8 a8 bes4 
    bes4 g2 
    f4 g8 a8 bes4 
    bes4 a4 f4 
    f4 f4. g8     %5
    a4 bes8 a8 bes4 
    ees,4 f8 ees8 f8 e8 
    f2 } f4 
  f2 f4 
  ees4. g8 ees8 d8   %10
  ees4 aes4 g4 
  g2 g4 
  f4 g4 a4 
  bes4. bes8 bes4 
  bes4 bes4 a4   %15
  f2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes4 f'8 ees8 f4 
    ees4 bes8 a8 bes8 c8 
    d2 g4 
    f4. ees8 d4 
    c8 bes8 c8 a8 bes4     %5
    c4 f,4 f4 
    c'4 bes4 f8 g8 
    a2 } c4 
  f,4 a4 bes4 
  bes4. bes8 c4~   %10
  c8 c8 c4 b4 
  c2 c4 
  c2 c4 
  d4. d8 ees4 
  bes4 f'4 ees8 d8   %15
  d2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    d,4 c4 bes4 
    ees4 g8 f8 g8 ees8 
    bes'8 a8 g4 ees4 
    f2 bes,4 
    a'8 g8 a8 f8 bes4     %5
    f4 ees4 d4 
    c4 d8 c8 d8 bes8 
    f2 } f'8 ees8 
  d4 ees8 d8 c8 bes8 
  ees4. ees8 aes4~   %10
  aes8 g8 f4 g4 
  c,2 ees4 
  f2 ees4 
  d4 g8 f8 ees8 d8 
  ees4 f4 f,4   %15
  bes2. 
  
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